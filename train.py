#-*- coding: utf-8 -*-
import argparse
import os
import sys
import time

import random
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm
import datasets
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

#from warprnnt_pytorch import RNNTLoss 
## Use it
if torch.__version__ == '1.12.0+cu116' or torch.__version__ == '1.12.1+cu116':
    from torchaudio.functional import rnnt_loss
else:
    from warp_rnnt import rnnt_loss
from train_optuna_v2 import recognize

import torch.distributed as dist

from utils import hparams as hp
from utils.utils import log_config, fill_variables, adjust_learning_rate, load_model, create_masks, init_weight, frame_stacking, get_learning_rate
from Models.transformer import Transformer

random.seed(77)
torch.random.manual_seed(777)
torch.cuda.manual_seed_all(777)
np.random.seed(777)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    if hp.optimizer_type == 'Noam' or hp.optimizer_type == 'AdamW':
        if epoch >= hp.decay_epoch:
            lr = adjust_learning_rate(optimizer, epoch, hp.decay_epoch)
        else:
            lr = get_learning_rate(step//hp.accum_grad+1, warmup_step=hp.warmup_step, warmup_factor=hp.warmup_factor, d_model=hp.d_model_e)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    else:
        lr = 1e-3

    optimizer.zero_grad()
    for d in dataloader:
        if hp.dev_mode:
            text, mel_input, pos_text, pos_mel, text_lengths, mel_lengths, real_flag = d
            real_flag = real_flag.to(device, non_blocking=True)
        else:
            text, mel_input, pos_text, pos_mel, text_lengths, mel_lengths = d

        text = text.to(device, non_blocking=True)
        mel_input = mel_input.to(device, non_blocking=True)
        pos_text = pos_text.to(device, non_blocking=True)
        pos_mel = pos_mel.to(device, non_blocking=True)
        text_lengths = text_lengths.to(device, non_blocking=True)
    
        if hp.frame_stacking > 1:
            mel_input, pos_mel = frame_stacking(mel_input, pos_mel, hp.frame_stacking)

        batch_size = mel_input.shape[0]
    
        if hp.decoder == 'LSTM' or hp.decoder.lower() == 'transducer':
            text_input = text
            src_mask, trg_mask = create_masks(pos_mel, pos_text)
        else:
            text_input = text[:, :-1]
            src_mask, trg_mask = create_masks(pos_mel, pos_text[:, :-1])

        with torch.cuda.amp.autocast(hp.amp): #and torch.autograd.set_detect_anomaly(True):
            if args.n_gpus > 1:
                dist.barrier()
            if hp.dev_mode:
                youtputs, ctc_outputs, attn_enc_enc, attn_dec_dec, attn_dec_enc = model(mel_input, text_input, src_mask, trg_mask, real_flag)
            else:
                youtputs, ctc_outputs, attn_enc_enc, attn_dec_dec, attn_dec_enc, iter_preds = model(mel_input, text_input, src_mask, trg_mask)
    
            print('step {} {}'.format(step, train_len))
            print('batch size = {}'.format(batch_size))
            print('lr = {}'.format(lr))

            if hp.use_lm_loss:
                lm_outputs = youtputs[1]
                youtputs = youtputs[0]
                
            if hp.decoder == 'LSTM' or hp.decoder.lower() == 'transformer':

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
                    onehot = torch.zeros((B * T, L), dtype=torch.float).to(DEVICE).scatter_(1, ys, 1)
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
                    loss_att /= batch_size
                else:
                    ys = text[:, 1:].contiguous().view(-1)
                    loss_att = F.cross_entropy(youtputs.view(-1, youtputs.size(-1)), ys, ignore_index=trg_pad)
                print('loss_att =', loss_att.item())
                loss = loss_att

            elif hp.decoder.lower() == 'transducer':
                mel_lengths_downsample = mel_lengths 
                for i in range(int(np.log2(hp.subsampling_rate))):
                    mel_lengths_downsample = (mel_lengths_downsample - 2) // 2
                transducer_outputs = F.log_softmax(youtputs, dim=-1)
                #loss_transducer = rnnt_loss(transducer_outputs, text.int(), mel_lengths_downsample.int(), text_lengths.int(), blank=0)
                #loss_transducer = rnnt_loss(transducer_outputs, text.int(), mel_lengths_downsample.to(device).int(), text_lengths.int(), average_frames=True, blank=0)

                loss_transducer = rnnt_loss(transducer_outputs, text.int(), mel_lengths_downsample.to(device).int(), text_lengths.int(), blank=0, reduction='mean')

                print(f'loss_transducer = {loss_transducer.item()}')
                loss = loss_transducer


            elif hp.decoder == 'ctc':
                predict_ts_ctc = F.log_softmax(ctc_outputs, dim=2).transpose(0, 1)
                mel_lengths_downsample = mel_lengths 
                
                for i in range(int(np.log2(hp.subsampling_rate))):
                    mel_lengths_downsample = (mel_lengths_downsample - 2) // 2
                loss_ctc = F.ctc_loss(predict_ts_ctc, text, mel_lengths_downsample, text_lengths, blank=0, zero_infinity=False)
                print('loss_ctc = {}'.format(loss_ctc.item()))
                print('zero')
                loss = loss_ctc

            if hp.use_ctc:
                predict_ts_ctc = F.log_softmax(ctc_outputs, dim=2).transpose(0, 1)
                #mel_lengths_downsample = ((mel_lengths - 2) // 2 - 2) // 2
                mel_lengths_downsample = mel_lengths 
                if hp.subsampling_rate > 1:
                    for i in range(int(np.log2(hp.subsampling_rate))):
                        mel_lengths_downsample = (mel_lengths_downsample - 2) // 2
                elif hp.frame_stacking > 1:
                    mel_lengths_downsample = mel_lengths_downsample // hp.frame_stacking
                loss_ctc = F.ctc_loss(predict_ts_ctc, text, mel_lengths_downsample, text_lengths, blank=0)
                print('loss_ctc = {}'.format(loss_ctc.item()))
                if not torch.isinf(loss_ctc):
                    loss = (hp.mtl_weight * loss + (1 - hp.mtl_weight) * loss_ctc)

            if hp.use_lm_loss:
                loss_lm = 0.0
                if hp.decoder == 'transducer':
                    ys = text.contiguous().view(-1, 1)
                else:
                    ys = text[:, 1:].contiguous().view(-1, 1)
                B, T, L = lm_outputs.shape
                eps = hp.eps_lm_loss
                log_prob = F.log_softmax(lm_outputs, dim=2)
                onehot = torch.zeros((B * T, L), dtype=torch.float).to(DEVICE).scatter_(1, ys, 1)
                onehot = onehot * (1 - 0.1) + (1 - onehot) * 0.1 / (youtputs.size(2) - 1)
                onehot = onehot.reshape(B, T, L)
                for i, t in enumerate(text_lengths):
                    if hp.decoder == 'transducer':
                        len_t = t - 1
                    else:
                        len_t = t
                    loss_lm += -(onehot[i, :len_t, :] * log_prob[i, :len_t, :]).sum() / len_t
                loss_lm /= batch_size
                
                print(f'loss_lm = {loss_lm}')
                loss += 0.5 * loss_lm

            ## iterative loss
            if len(hp.iter_loss) != 0:
                loss_ctc_iter = 0
                for iter_pred in iter_preds:
                    loss_ctc_iter += F.ctc_loss(predict_ts_ctc, text, mel_lengths_downsample, text_lengths, blank=0)
                print(f'loss_ctc_iter = {loss_ctc_iter.item()}')
                if not torch.isinf(loss_ctc_iter):
                    loss += 0.3 * loss_ctc_iter

            step += 1
            print('loss =', loss.item())
        if not torch.isnan(loss):
            if hp.amp:
                loss /= hp.accum_grad
                scaler.scale(loss).backward()
                if args.debug:
                    print('debug???')
                    average_gradients(model)
                #scaler.unscale_(optimizer)
                #torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
                #scaler.step(optimizer)
                #scaler.update()
                #print(f'backward {time.time() - local_time}')
                #local_time = time.time()
                if step % hp.accum_grad == 0:
                    #scaler.unscale_(optimizer)
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
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

            if step % hp.accum_grad == 0 and (hp.optimizer_type == 'Noam' or hp.optimizer_type == 'AdamW'):
                if epoch < hp.decay_epoch:
                    lr = get_learning_rate(step//hp.accum_grad+1, warmup_step=hp.warmup_step, warmup_factor=hp.warmup_factor, d_model=hp.d_model_e)

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
        else:
            print('loss is nan')
            del loss
            #load_dir = hp.save_dir
            #map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            #loaded_dict = load_model("{}".format(os.path.join(load_dir, 'network.epoch{}'.format(epoch))), map_location=map_location)
            #model.load_state_dict(loaded_dict)
            sys.exit(1)
        if step % hp.accum_grad == 0 and (hp.optimizer_type == 'Noam' or hp.optimizer_type == 'AdamW'):
            optimizer.zero_grad()
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
        #torch.save(model.to('cpu').state_dict(), hp.save_dir+"/network.epoch{}".format(epoch+1))
        torch.save(model.state_dict(), hp.save_dir+"/network.epoch{}".format(epoch+1))
        print('save model')
        #model = model.to(rank)
        #torch.save(model.state_dict(), hp.save_dir+"/network.epoch{}".format(epoch+1))
    
    if rank == 0 and (epoch + 1) % hp.save_per_epoch == 0:
        torch.save(optimizer.state_dict(), hp.save_dir+"/network.optimizer.epoch{}".format(epoch+1))
        print('save optimizer')

    if args.n_gpus > 1:
        dist.barrier()
    return step


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


def train_epoch(model, optimizer, args, hp, step, start_epoch=0, rank=0):
    dataloader = get_dataloader(step, args, hp)

    for epoch in range(start_epoch, hp.max_epoch):
        start_time = time.time()
        if step // hp.accum_grad > hp.warmup_step:
            dataloader = get_dataloader(step, args, hp)

        step = train_loop(model, optimizer, step, epoch, args, hp, rank, dataloader)
 
        print("EPOCH {} end".format(epoch+1))
        print('elapsed time = {}'.format(time.time() - start_time))
        if hp.dev_script is not None:
            if (epoch >= 20 and epoch % 5 == 0) or epoch == 1 and rank == 0:
                print('eval')
                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                loaded_dict = load_model("{}".format(os.path.join(hp.save_dir, 'network.epoch{}'.format(epoch+1))), map_location=map_location)
                model_dev = Transformer(hp)
                model_dev.load_state_dict(loaded_dict)
                model_dev.eval()
                model_dev.to(rank)
                wer_all = recognize(hp, model_dev, model_lm=None, lm_weight=0.0, calc_wer=True, rank=rank)
                print(f'{args.hp_file} epoch {epoch} wer is {wer_all}')

                del model_dev
        #if args.n_gpus > 1:
        #    dist.barrier()


def init_distributed(rank, n_gpus, port):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."

    torch.cuda.set_device(rank % n_gpus)

    os.environ['MASTER_ADDR'] = 'localhost' #dist_config.MASTER_ADDR
    os.environ['MASTER_PORT'] = port #dist_config.MASTER_PORT

    torch.distributed.init_process_group(
        backend='nccl', world_size=n_gpus, rank=rank
    )


def cleanup():
    torch.distributed.destroy_process_group()


def run_distributed(fn, args, hp):
    port = '60' + str(int(time.time()))[-4:]
    #port = str(np.random.randint(600000, 610000)) 
    print(f'port = {port}')
    try:
        mp.spawn(fn, args=(args, hp, port), nprocs=args.n_gpus, join=True)
    except:
        cleanup()

def run_training(rank, args, hp, port=None):
    if args.n_gpus > 1:
        init_distributed(rank, args.n_gpus, port)
        torch.cuda.set_device(f'cuda:{rank}')

    model = Transformer(hp)
    #model.apply(init_weight)
    model.train()

    if rank == 0:
        print(model)

    model = model.to(rank)

    #print(model)
    if args.n_gpus > 1:
        model = DDP(torch.nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[rank])
    
    max_lr = hp.init_lr
    if hp.optimizer_type == 'Noam':
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
    elif hp.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
        print(optimizer)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    
    assert (hp.batch_size is None) != (hp.max_seqlen is None)

    if args.n_gpus > 1:
        dist.barrier()
        # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

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
            #train_dataset = datasets.get_dataset(hp.train_script, hp, spec_aug=False, feat_norm=[hp.mean_file, hp.var_file])
            #collate_fn_transformer = datasets.collate_fn
            #if hp.batch_size is not None:
            #    sampler = datasets.NumBatchSampler(train_dataset, hp.batch_size)
            #elif hp.max_seqlen is not None:
            #    sampler = datasets.LengthsBatchSampler(train_dataset, hp.max_seqlen, hp.lengths_file, shuffle=True, shuffle_one_time=False, shuffle_all=hp.dataset_shuffle_all)
            #train_sampler = DistributedSampler(sampler) if args.n_gpus > 1 else sampler
            #dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, collate_fn=collate_fn_transformer)
            loaded_dict = torch.load("{}".format(os.path.join(load_dir, 'network.optimizer.epoch{}'.format(hp.loaded_epoch))), map_location=map_location)
            optimizer.load_state_dict(loaded_dict)
            step = loaded_dict['state'][0]['step'] * hp.accum_grad
            lr = get_learning_rate(step//hp.accum_grad+1, warmup_step=hp.warmup_step, warmup_factor=hp.warmup_factor, d_model=hp.d_model_e)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            del loaded_dict
            torch.cuda.empty_cache()
    else:
        start_epoch = 0
        step = 1
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('params = {0:.2f}M'.format(pytorch_total_params / 1000 / 1000))
    train_epoch(model, optimizer, args, hp, step=step, start_epoch=start_epoch, rank=rank)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', type=str, default='hparams.py')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    hp.configure(args.hp_file)
    fill_variables(hp)
    log_config(hp)

    os.makedirs(hp.save_dir, exist_ok=True)

    # # multi-gpu setup
    # if torch.cuda.device_count() > 1:
    #     # multi-gpu configuration
    #     ngpu = torch.cuda.device_count()
    #     device_ids = list(range(ngpu))
    #     model = torch.nn.DataParallel(model, device_ids)
    #     model.cuda()
    # else:
    #     model.to(DEVICE)
    
    n_gpus = torch.cuda.device_count()
    args.__setattr__('n_gpus', n_gpus)

    if n_gpus > 1:
        run_distributed(run_training, args, hp)
    else:
        run_training(0, args, hp, None)
