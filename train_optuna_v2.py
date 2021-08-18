# -*- coding: utf-8 -*-
# test comment
import argparse
import os
import sys
import time

import copy
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
import datasets
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import sentencepiece as spm

import utils
from utils.utils import log_config, fill_variables, adjust_learning_rate, load_model, create_masks, init_weight, frame_stacking, average_checkpoints, load_dat
from Models.transformer import Transformer

from tools.calc_wer import wer

import mlflow
import optuna
from optuna.trial import TrialState

random.seed(77)
torch.random.manual_seed(777)
torch.cuda.manual_seed_all(777)
np.random.seed(777)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_learning_rate(step, hp):
    warmup_step = hp.warmup_step  # 4000
    warmup_factor = hp.warmup_factor #5.0
    d_model = hp.d_model_e
    return warmup_factor * min(step ** -0.5, step * warmup_step ** -1.5) * (d_model ** -0.5)


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def train_loop(model, optimizer, step, epoch, args, hp, rank, dataloader):
    scaler = torch.cuda.amp.GradScaler()
    src_pad = 0
    trg_pad = 0

    train_len = len(dataloader)
    local_time = time.time()
    device = f'cuda:{rank}'
    label_smoothing = True
    if hp.optimizer_type == 'Noam':
        if epoch >= hp.decay_epoch:
            lr = adjust_learning_rate(optimizer, epoch, hp.decay_epoch)
        else:
            lr = get_learning_rate(step//hp.accum_grad+1, hp)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    optimizer.zero_grad()
    loss_avg_att = 0
    loss_avg_ctc = 0
    num_samples = 0
    is_nan = False
    # with tqdm(dataloader) as pbar:
    if True:
        for d in dataloader:
            text, mel_input, pos_text, pos_mel, text_lengths, mel_lengths = d

            text = text.to(device, non_blocking=True)
            mel_input = mel_input.to(device, non_blocking=True)
            pos_text = pos_text.to(device, non_blocking=True)
            pos_mel = pos_mel.to(device, non_blocking=True)
            text_lengths = text_lengths.to(device, non_blocking=True)
        
            #if hp.frame_stacking is not None:
            #    mel_input, pos_mel = frame_stacking(mel_input, pos_mel, hp.frame_stacking)

            batch_size = mel_input.shape[0]

            if hp.decoder == 'LSTM':
                text_input = text
                src_mask, trg_mask = create_masks(pos_mel, pos_text)
            else:
                text_input = text[:, :-1]
                src_mask, trg_mask = create_masks(pos_mel, pos_text[:, :-1])

            with torch.cuda.amp.autocast(hp.amp):
                if args.n_gpus > 1:
                    dist.barrier()
                youtputs, ctc_outputs, attn_enc_enc, attn_dec_dec, attn_dec_enc = model(mel_input, text_input, src_mask, trg_mask)

                if hp.decoder != 'ctc':
                    loss_att = 0.0
                    # cross entropy
                    if label_smoothing:
                        if hp.decoder == 'LSTM':
                            ys = text.contiguous().view(-1, 1)
                        else:
                            ys = text[:, 1:].contiguous().view(-1, 1)
                        B, T, L = youtputs.shape
                        #eps = 0.1
                        log_prob = F.log_softmax(youtputs, dim=2)
                        onehot = torch.zeros((B * T, L), dtype=torch.float).to(device).scatter_(1, ys, 1)
                        onehot = onehot * (1 - 0.1) + (1 - onehot) * 0.1 / (youtputs.size(2) - 1)
                        onehot = onehot.reshape(B, T, L)
                        for i, t in enumerate(text_lengths):
                            if hp.decoder == 'LSTM':
                                len_t = t
                            else:
                                len_t = t - 1
                            if hp.T_norm:
                                loss_att += -(onehot[i, :len_t, :] * log_prob[i, :len_t, :]).sum() / len_t
                            else:
                                loss_att += -(onehot[i, :len_t, :] * log_prob[i, :len_t, :]).sum()
                        loss_avg_att += loss_att.item()
                        num_samples += batch_size
                        loss_att /= batch_size
                    else:
                        ys = text[:, 1:].contiguous().view(-1)
                        loss_att = F.cross_entropy(youtputs.view(-1, youtputs.size(-1)), ys, ignore_index=trg_pad)
                    print('loss_att =', loss_att.item())

                print('step {} {}'.format(step, train_len))
                print('batch size = {}'.format(batch_size))
                print('lr = {}'.format(lr))
                step += 1
 
                if hp.decoder == 'ctc':
                    predict_ts_ctc = F.log_softmax(ctc_outputs, dim=2).transpose(0, 1)
                    mel_lengths_downsample = mel_lengths 
                    for i in range(int(np.log2(hp.subsampling_rate))):
                        mel_lengths_downsample = (mel_lengths_downsample - 2) // 2
                    loss_ctc = F.ctc_loss(predict_ts_ctc, text, mel_lengths_downsample, text_lengths, blank=0, zero_infinity=False)
                    #print('loss_ctc = {}'.format(loss_ctc.item()))
                    loss = loss_ctc

                elif hp.mode == 'ctc-transformer':
                    predict_ts_ctc = F.log_softmax(ctc_outputs, dim=2).transpose(0, 1)
                    #mel_lengths_downsample = ((mel_lengths - 2) // 2 - 2) // 2
                    mel_lengths_downsample = mel_lengths 
                    for i in range(int(np.log2(hp.subsampling_rate))):
                        mel_lengths_downsample = (mel_lengths_downsample - 2) // 2
                    loss_ctc = F.ctc_loss(predict_ts_ctc, text, mel_lengths_downsample, text_lengths, blank=0)
                    print('loss_ctc = {}'.format(loss_ctc.item()))
                    loss_avg_ctc += loss_ctc.item() * batch_size
                    loss = (hp.mlt_weight * loss_att + (1 - hp.mlt_weight) * loss_ctc)
                else:
                    loss = loss_att
                print('loss =', loss.item())
                if not torch.isnan(loss):
                    if hp.amp:
                        loss /= hp.accum_grad
                        scaler.scale(loss).backward()
                        #if args.debug:
                        #    print('debug???')
                        #    average_gradients(model)
                        #scaler.unscale_(optimizer)
                        #torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
                        #scaler.step(optimizer)
                        #scaler.update()
                        #print(f'backward {time.time() - local_time}')
                        #local_time = time.time()
                        if step % hp.accum_grad == 0:
                            if hp.clip is not None:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        loss /= hp.accum_grad
                        loss.backward()
                        if step % hp.accum_grad == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
                            optimizer.step()

                    if step % hp.accum_grad == 0 and hp.optimizer_type == 'Noam':
                        lr = get_learning_rate(step // hp.accum_grad + 1, hp)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                else:
                    print('loss is nan')
                    is_nan = True
                    break
                if step % hp.accum_grad == 0 and hp.optimizer_type == 'Noam':
                    optimizer.zero_grad()
                #pbar.set_description(f'loss = {loss.item()}')
                sys.stdout.flush()
                # calc
    #n_correct = 0
    #for i, t in enumerate(text_lengths):
    #    tmp = youtputs[i, :t-1, :].max(1)[1].cpu().numpy()
    #    for j in range(t-1):
    #        if tmp[j] == text[i][j+1]:
    #            n_correct = n_correct + 1
    #acc = 1.0 * n_correct / float(sum(text_lengths))
    #print('acc = {}'.format(acc))

    if rank == 0 and ((epoch+1) % hp.save_per_epoch >= (hp.save_per_epoch - 10) or (epoch+1) % hp.save_per_epoch == 0):
        torch.save(model.state_dict(), hp.save_dir+"/network.epoch{}".format(epoch+1))
        print('save model')
    if rank==0 and (epoch+1) % hp.save_per_epoch == 0:
        torch.save(optimizer.state_dict(), hp.save_dir+"/network.optimizer.epoch{}".format(epoch+1))
        print('save optimizer')

    if args.n_gpus > 1:
        dist.barrier()
    return step, is_nan, loss_avg_att / num_samples, loss_avg_ctc / num_samples


def get_dataloader(step, args, hp):
    if step // hp.accum_grad > hp.warmup_step:
        train_dataset = datasets.get_dataset(hp.train_script, hp, spec_aug=hp.use_spec_aug, feat_norm=[hp.mean_file, hp.var_file])
    else:
        train_dataset = datasets.get_dataset(hp.train_script, hp, spec_aug=False, feat_norm=[hp.mean_file, hp.var_file])

    collate_fn_transformer = datasets.collate_fn
    if hp.batch_size is not None:
        sampler = datasets.NumBatchSampler(train_dataset, hp.batch_size)
    elif hp.max_seqlen is not None:
        sampler = datasets.LengthsBatchSampler(train_dataset, hp.max_seqlen, hp.lengths_file, shuffle=True, shuffle_one_time=False, shuffle_all=hp.dataset_shuffle_all)

    train_sampler = datasets.DistributedSamplerWrapper(sampler) if args.n_gpus > 1 else sampler
    dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=8, collate_fn=collate_fn_transformer)

    return dataloader

#def objective(single_trial):
def objective(trial):
    rank = 0 #dist.get_rank()
    device = f'cuda:{rank}'
    #torch.cuda.set_device(f'cuda:{rank}')
    #trial = optuna.integration.TorchDistributedTrial(single_trial, device=device)
    hparams_dir = args.base_dir
    candidates = []
    for i in range(1, args.n_trials+1):
        hparams_path = os.path.join(hparams_dir, f'hparams{i}.py')
        candidates.append(hparams_path)

    print(candidates)
    hp_file = trial.suggest_categorical("hp_file", candidates)

    print(hp_file)
    hp = utils.HParams()
    hp.configure(hp_file)
    fill_variables(hp, verbose=False)
    #log_config(hp)

    #if args.n_gpus > 1:
    #    init_distributed(rank, args.n_gpus)
    #    torch.cuda.set_device(f'cuda:{rank}')

    model = Transformer(hp)
    model.apply(init_weight)
    model.train()

    model = model.to(rank)
    if args.n_gpus > 1:
        model = DDP(torch.nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[rank])
    #model = DDP(torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.to(device)))
    
    max_lr = hp.init_lr
    if args.n_gpus > 1:
        dist.barrier()
        # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if hp.optimizer_type == 'Noam':
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    
    assert (hp.batch_size is None) != (hp.max_seqlen is None)
    if hp.loaded_epoch is not None:
        start_epoch = hp.loaded_epoch
        load_dir = hp.loaded_dir
        print('epoch {} loaded'.format(hp.loaded_epoch))
        loaded_dict = load_model("{}".format(os.path.join(load_dir, 'network.epoch{}'.format(hp.loaded_epoch))), map_location=map_location)
        model.load_state_dict(loaded_dict)
        if hp.is_flat_start:
            step = 1
            start_epoch = 0
            print('flat_start')
        else:
            loaded_dict = torch.load("{}".format(os.path.join(load_dir, 'network.optimizer.epoch{}'.format(hp.loaded_epoch))), map_location=map_location)
            optimizer.load_state_dict(loaded_dict)
            step = loaded_dict['state'][0]['step'] * hp.accum_grad
            lr = get_learning_rate(step//hp.accum_grad+1, hp)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            del loaded_dict
            torch.cuda.empty_cache()
    else:
        start_epoch = 0
        step = 1

    #step = 1
    #pytorch_total_params = sum(p.numel() for p in model.parameters())
    #print('params = {0:.2f}M'.format(pytorch_total_params / 1000 / 1000))
    dataloader = get_dataloader(step, args, hp)
    prune_threshold = {'20': 0.158}
    prune_flag = False
    for epoch in range(start_epoch, hp.max_epoch+1):
        start_time = time.time()
        if step // hp.accum_grad > hp.warmup_step:
            dataloader = get_dataloader(step, args, hp)
        step, is_nan, loss_avg_att, loss_avg_ctc = train_loop(model, optimizer, step, epoch, args, hp, rank, dataloader)
        print("EPOCH {} end".format(epoch+1))
        print(f'loss_avg_att = {loss_avg_att}')
        print(f'loss_avg_ctc = {loss_avg_ctc}')
        print('elapsed time = {}'.format(time.time() - start_time))

        if epoch >= 10 and epoch % 5 == 0:
            print('eval')
            #avg_states = average_checkpoints(epoch-9, epoch, hp)
            #model_dev = Transformer(hp)
            #model_dev.load_state_dict(avg_states)
            #model_dev.eval()

            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            loaded_dict = load_model("{}".format(os.path.join(hp.save_dir, 'network.epoch{}'.format(epoch+1))), map_location=map_location)
            model_dev = Transformer(hp)
            model_dev.load_state_dict(loaded_dict)
            model_dev.eval()
            model_dev.to(rank)
            wer_all = recognize(hp, model_dev, model_lm=None, lm_weight=0.0, calc_wer=True, rank=rank)
            print(f'{hp_file} epoch {epoch} wer is {wer_all}')

            del model_dev
            trial.report(wer_all, epoch)
            if str(epoch) in prune_threshold.keys():
                if prune_threshold[str(epoch)] < wer_all and args.task == 'librispeech': 
                    prune_flag = True

        if trial.should_prune() or is_nan is True or prune_flag:
            raise optuna.exceptions.TrialPruned()

    return wer_all


def recognize(hp, model_dev, model_lm, lm_weight, calc_wer=True, rank=None):
    device = f'cuda:{rank}'
    model_dev.to(device)
    sp = spm.SentencePieceProcessor()
    sp.Load(hp.spm_model)
    INIT_TOK = sp.bos_id()
    EOS_TOK = sp.eos_id()
    BATCH_SIZE = 1
    script_file = hp.dev_script
    script_buf = []
    with open(script_file) as f:
        for line in f:
            script_buf.append(line)
    
    num_mb = len(script_buf) // BATCH_SIZE
    results_all = np.zeros(5)
    
    for i in range(num_mb):
        xs = []
        lengths = []
        # ts_lengths = []
        for j in range(BATCH_SIZE):
            s = script_buf[i*BATCH_SIZE+j].strip()
            if len(s.split('|')) == 1:
                x_file = s
            else:
                x_file, laborg = s.split('|', 1)
            if '.htk' in x_file:
                cpudat = load_dat(x_file)
                cpudat = cpudat[:, :hp.mel_dim]
                if hp.mean_utt:
                     cpudat = cpudat - cpudat.mean(axis=0, keepdims=True)
                if hp.mean_file is not None and hp.var_file is not None:
                    mean = np.load(hp.mean_file).reshape(1, -1)
                    var = np.load(hp.var_file).reshape(1, -1)
                    cpudat = (cpudat - mean) / np.sqrt(var)
            elif '.npy' in x_file:
                cpudat = np.load(x_file)
            lengths.append(cpudat.shape[0])
            xs.append(cpudat)
    
        xs_dummy = []
        src_pad = 0
        for i in range(len(xs)):
            xs_dummy.append([1] * lengths[i])
        src_seq = np.zeros((BATCH_SIZE, max(lengths), hp.mel_dim))
        for i in range(len(xs)):
            src_seq[i, :lengths[i], :] = xs[i]
        src_seq_dummy = np.array([inst + [src_pad] * (max(lengths) - len(inst)) for inst in xs_dummy])
    
        src_seq = torch.from_numpy(src_seq).to(device).float()
        src_seq_dummy = torch.from_numpy(src_seq_dummy).to(device).long()
        youtput_in_Variable = model_dev.decode(src_seq, src_seq_dummy, 10, model_lm, INIT_TOK, EOS_TOK, lm_weight)
    
        if calc_wer:
            if len(youtput_in_Variable) == 0:
                results = "Dummy"
            else:
                results = "{}".format(sp.DecodeIds(youtput_in_Variable))
            results_all += wer(laborg.split(), results.split())

    wer_results_all = results_all[1:-1].sum() / results_all[-1]

    return wer_results_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--task', type=str, default=None)
    args = parser.parse_args()

    #os.makedirs(hp.save_dir, exist_ok=True)
    n_gpus = torch.cuda.device_count()
    args.__setattr__('n_gpus', n_gpus)

    #world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE")) if os.environ.get("OMPI_COMM_WORLD_SIZE") is not None else None
    #if world_size is None:
    #    world_size = 1 #int(os.environ.get("PMI_SIZE"))
    #if world_size is None:
    #    raise RuntimeError("Neither MPICH nor OpenMPI is avaliable.")
    #os.environ["WORLD_SIZE"] = str(world_size)

    #rank = int(os.environ.get("OMPI_COMM_WORLD_RANK")) if os.environ.get("OMPI_COMM_WORLD_RANK") is not None else None
    #if rank is None:
    #    rank = 0 #int(os.environ.get("PMI_RANK"))
    #os.environ["RANK"] = str(rank)

    #os.environ["MASTER_ADDR"] = "127.0.0.1"
    #os.environ["MASTER_PORT"] = "20001"

    ##local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    ##dist.init_process_group("gloo")
    #print('rank', rank)
    ##torch.cuda.set_device(f'cuda:{int(rank)}')
    #dist.init_process_group(
    #    backend='nccl', world_size=world_size, rank=rank
    #)
    #rank = dist.get_rank()

    hparams_dir = args.base_dir
    candidates = {"hp_file":[]}
    for i in range(1, args.n_trials+1):
        hparams_path = os.path.join(hparams_dir, f'hparams{i}.py')
        candidates["hp_file"].append(hparams_path)

    study = None
    n_trials = args.n_trials
    rank = 0
    if rank == 0:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(candidates), pruner=optuna.pruners.MedianPruner(
        n_startup_trials=2, n_warmup_steps=0, interval_steps=1))
        #study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
    else:
        for _ in range(n_trials):
            try:
                objective(None)
            except optuna.TrialPruned:
                pass

    if rank == 0:
        assert study is not None
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    
    #if n_gpus > 1:
    #    run_distributed(run_training, args)
    #else:
    #    run_training(0, args)
