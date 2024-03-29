# -*- coding: utf-8 -*-
# test comment
import argparse
import os
import sys
import time

import random
import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
import datasets_wav2vec2
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.distributed as dist

from utils import hparams as hp
from utils.utils import log_config, fill_variables, adjust_learning_rate, load_model, create_masks, init_weight
from Models.transformer_wav2vec2 import TransformerWav2vec2

random.seed(77)
torch.random.manual_seed(777)
torch.cuda.manual_seed_all(777)
np.random.seed(777)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_learning_rate(step, hp):
    warmup_step = hp.warmup_step #hp.warmup_step # 4000
    warmup_factor = hp.warmup_factor #hp.warmup_factor #10.0 # 1.0
    d_model = 256
    return warmup_factor * min(step ** -0.5, step * warmup_step ** -1.5) * (d_model ** -0.5)

def get_learning_rate_tristage(step):
    max_update = 80000
    phase_ratio = [0.1, 0.4, 0.5]
    final_lr_scale = 0.05
    init_lr_scale = 0.01

    peak_lr = 0.00003
    init_lr = init_lr_scale * peak_lr
    final_lr = final_lr_scale * peak_lr
    
    warmup_steps = int(max_update * phase_ratio[0])
    hold_steps = int(max_update * phase_ratio[1])
    decay_steps = int(max_update * phase_ratio[2])

    warmup_rate = (peak_lr - init_lr) / warmup_steps
    decay_factor = -math.log(final_lr_scale) / decay_steps

    # stage 0
    if step < warmup_steps:
        lr = init_lr + warmup_rate * step
        return lr
    
    offset = warmup_steps
    # stage 1
    if step < offset + hold_steps:
        lr = peak_lr
        return lr

    offset += hold_steps
    # stage 2
    if step <= offset + decay_steps:
        lr = peak_lr *  math.exp(-decay_factor * step)
        return lr

    # stage 3
    return final_lr

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
            if hp.lr_tristage:
                lr = get_learning_rate_tristage(step // hp.accum_grad + 1)
            else:
                lr = get_learning_rate(step//hp.accum_grad+1, hp)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    optimizer.zero_grad()
    for d in dataloader:
        text, wav_input, pos_text, pos_wav2vec2, text_lengths, wav2vec2_lengths = d

        text = text.to(device, non_blocking=True)
        wav_input = wav_input.to(device, non_blocking=True)
        pos_text = pos_text.to(device, non_blocking=True)
        pos_wav2vec2 = pos_wav2vec2.to(device, non_blocking=True)
        text_lengths = text_lengths.to(device, non_blocking=True)
    
        batch_size = wav_input.shape[0]
    
        if hp.decoder == 'LSTM':
            text_input = text
            src_mask, trg_mask = create_masks(pos_wav2vec2, pos_text)
        else:
            text_input = text[:, :-1]
            src_mask, trg_mask = create_masks(pos_wav2vec2, pos_text[:, :-1])

        with torch.cuda.amp.autocast(hp.amp): #and torch.autograd.set_detect_anomaly(True):
            if args.n_gpus > 1:
                dist.barrier()
            youtputs, ctc_outputs, attn_dec_dec, attn_dec_enc = model(wav_input, text_input, src_mask, trg_mask, step//hp.accum_grad+1)

            print('step {} {}'.format(step, train_len))
            print('batch size = {}'.format(batch_size))
            print('lr = {}'.format(lr))
            step += 1
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
                print(f'loss_att = {loss_att.item()}')


            if hp.decoder == 'ctc':
                predict_ts_ctc = F.log_softmax(ctc_outputs, dim=2).transpose(0, 1)
                loss_ctc = F.ctc_loss(predict_ts_ctc, text, wav2vec2_lengths, text_lengths, blank=0)
                print('loss_ctc = {}'.format(loss_ctc.item()))
                loss = loss_ctc
            elif hp.use_ctc:
                ## NOTE: ctc loss does not support fp16?
                predict_ts_ctc = F.log_softmax(ctc_outputs, dim=2).transpose(0, 1)
                loss_ctc = F.ctc_loss(predict_ts_ctc, text, wav2vec2_lengths, text_lengths, blank=0)
                print('loss_ctc = {}'.format(loss_ctc.item()))
                loss = (hp.mlt_weight * loss_att + (1 - hp.mlt_weight) * loss_ctc)
            else:            
                loss = loss_att
            print('loss =', loss.item())
        if not torch.isnan(loss):
            if hp.amp:
                loss /= hp.accum_grad
                scaler.scale(loss).backward()
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
                    optimizer.step()

            if step % hp.accum_grad == 0 and hp.optimizer_type == 'Noam':
                if epoch < hp.decay_epoch:
                    if hp.lr_tristage:
                        lr = get_learning_rate_tristage(step // hp.accum_grad + 1)
                    else:
                        lr = get_learning_rate(step // hp.accum_grad + 1, hp)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
        else:
            print('loss is nan')
            del loss
            sys.exit(1)
        if step % hp.accum_grad == 0 and hp.optimizer_type == 'Noam':
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
        torch.save(model.state_dict(), hp.save_dir+"/network.epoch{}".format(epoch+1))
        print('save model')
    
    if rank == 0 and (epoch + 1) % hp.save_per_epoch == 0:
        torch.save(optimizer.state_dict(), hp.save_dir+"/network.optimizer.epoch{}".format(epoch+1))
        print('save optimizer')

    #if args.n_gpus > 1:
    #    dist.barrier()
    return step


def get_dataloader(step, args, hp):
    ## TODO: Mask setting
    if step // hp.accum_grad > hp.warmup_step:
        train_dataset = datasets_wav2vec2.TrainDatasets(hp.train_script, hp)
    else:
        train_dataset = datasets_wav2vec2.TrainDatasets(hp.train_script, hp)

    collate_fn_transformer = datasets_wav2vec2.collate_fn
    if hp.batch_size is not None:
        sampler = datasets_wav2vec2.NumBatchSampler(train_dataset, hp.batch_size)
    elif hp.max_seqlen is not None:
        sampler = datasets_wav2vec2.LengthsBatchSampler(train_dataset, hp.max_seqlen, hp.lengths_file, shuffle=True, shuffle_one_time=False, shuffle_all=hp.dataset_shuffle_all)

    train_sampler = datasets_wav2vec2.DistributedSamplerWrapper(sampler) if args.n_gpus > 1 else sampler
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
    print(f'port = {port}')
    try:
        mp.spawn(fn, args=(args, hp, port), nprocs=args.n_gpus, join=True)
    except:
        cleanup()

def run_training(rank, args, hp, port=None):
    if args.n_gpus > 1:
        init_distributed(rank, args.n_gpus, port)
        torch.cuda.set_device(f'cuda:{rank}')

    ## NOTE: variable
    model = TransformerWav2vec2(hp, pretrain_model='facebook/wav2vec2-large-lv60', freeze_feature_extractor=hp.freeze_feature_extractor)
    ## TODO: change init_weight (maybe initialize all networks)
    #model.apply(init_weight)
    model.train()

    if rank == 0:
        print(model)

    model = model.to(rank)

    if args.n_gpus > 1:
        model = DDP(torch.nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[rank])
    
    max_lr = hp.init_lr
    if hp.optimizer_type == 'Noam':
        ## NOTE: scheduling?
        ## NOTE: learning rate?
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
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
            loaded_dict = torch.load("{}".format(os.path.join(load_dir, 'network.optimizer.epoch{}'.format(hp.loaded_epoch))), map_location=map_location)
            optimizer.load_state_dict(loaded_dict)
            step = loaded_dict['state'][0]['step']
            #lr = get_learning_rate(step//hp.accum_grad+1, hp)
            lr = get_learning_rate_tristage(step // hp.accum_grad + 1)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', type=str, default='hparams.py')
    args = parser.parse_args()
    hp.configure(args.hp_file)
    fill_variables(hp)
    log_config(hp)

    os.makedirs(hp.save_dir, exist_ok=True)

    n_gpus = torch.cuda.device_count()
    args.__setattr__('n_gpus', n_gpus)

    if n_gpus > 1:
        run_distributed(run_training, args, hp)
    else:
        run_training(0, args, hp, None)

if __name__ == '__main__':
    main()
