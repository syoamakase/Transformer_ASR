# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time

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

from utils import hparams as hp
from utils.utils import log_config, fill_variables, adjust_learning_rate, load_model, create_masks, init_weight
from Models.transformer import Transformer

random.seed(77)
torch.random.manual_seed(777)
torch.cuda.manual_seed_all(777)
np.random.seed(777)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_learning_rate(step):
    warmup_step = 25000 #hp.warmup_step # 4000
    warmup_factor = 5.0 #hp.warmup_factor #10.0 # 1.0
    d_model = 256
    return warmup_factor * min(step ** -0.5, step * warmup_step ** -1.5) * (d_model ** -0.5)

def train_loop(model, optimizer, step, epoch, args, hp, rank):
    scaler = torch.cuda.amp.GradScaler()
    src_pad = 0
    trg_pad = 0
    if step > hp.warmup_step:
        train_dataset = datasets.get_dataset(hp.train_script, hp.spm_model, hp, spec_aug=hp.use_spec_aug, feat_norm=[hp.mean_file, hp.var_file])
    else:
        train_dataset = datasets.get_dataset(hp.train_script, hp.spm_model, hp, spec_aug=False, feat_norm=[hp.mean_file, hp.var_file])

    collate_fn_transformer = datasets.collate_fn
    if hp.batch_size is not None:
        sampler = datasets.NumBatchSampler(train_dataset, hp.batch_size)
    elif hp.max_seqlen is not None:
        sampler = datasets.LengthsBatchSampler(train_dataset, hp.max_seqlen, hp.lengths_file, shuffle=True, shuffle_one_time=False, shuffle_all=hp.dataset_shuffle_all)

    train_sampler = datasets.DistributedSamplerWrapper(sampler) if args.n_gpus > 1 else sampler
    dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, collate_fn=collate_fn_transformer)

    train_len = len(dataloader)
    local_time = time.time()
    device = f'cuda:{rank}'
    label_smoothing = True
    for d in dataloader:
        if hp.optimizer_type == 'Noam':
            lr = get_learning_rate(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
 
        text, mel_input, pos_text, pos_mel, text_lengths, mel_lengths = d

        text = text.to(device, non_blocking=True)
        mel_input = mel_input.to(device, non_blocking=True)
        pos_text = pos_text.to(device, non_blocking=True)
        pos_mel = pos_mel.to(device, non_blocking=True)
        text_lengths = text_lengths.to(device, non_blocking=True)
    
        batch_size = mel_input.shape[0]
    
        text_input = text[:, :-1]
        src_mask, trg_mask = create_masks(pos_mel, pos_text[:, :-1])

        print(f'load {time.time() - local_time}')
        local_time = time.time()
    
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(hp.amp): #and torch.autograd.set_detect_anomaly(True):
            dist.barrier()
            if hp.mode == 'ctc-transformer':
                youtputs, ctc_outputs, attn_enc_enc, attn_dec_dec, attn_dec_enc = model(mel_input, text_input, src_mask, trg_mask)
            else:
                youtputs = model(mel_input, text_input, src_mask, trg_mask)
    
            print(f'forward {time.time() - local_time}')
            local_time = time.time()

            loss_att = 0.0
            # cross entropy
            if label_smoothing:
                ys = text[:, 1:].contiguous().view(-1)
                B, T, L = youtputs.shape
                eps = 0.1
                log_prob = F.log_softmax(youtputs, dim=2)
                onehot = torch.zeros((B * T, L), dtype=torch.float).to(DEVICE).scatter_(1, ys.reshape(-1, 1), 1)
                onehot = onehot * (1 - eps) + (1 - onehot) * eps / (youtputs.size(2) - 1)
                onehot = onehot.reshape(B, T, L)
                for i, t in enumerate(text_lengths):
                    if hp.T_norm:
                        loss_att += -(onehot[i, :t-1, :] * log_prob[i, :t-1, :]).sum() / (t-1)
                    else:
                        loss_att += -(onehot[i, :t-1, :] * log_prob[i, :t-1, :]).sum()
                loss_att /= batch_size
            else:
                ys = text[:,1:].contiguous().view(-1)
                loss_att = F.cross_entropy(youtputs.view(-1, youtputs.size(-1)), ys, ignore_index=trg_pad)

            print('step {} {}'.format(step, train_len))
            print('batch size = {}'.format(batch_size))
            print('lr = {}'.format(lr))
            step += 1
 
            print('loss_att =', loss_att.item())
            if hp.mode == 'ctc-transformer':
                predict_ts_ctc = F.log_softmax(ctc_outputs, dim=2).transpose(0, 1)
                mel_lengths_downsample = mel_lengths // 4 - 1
                loss_ctc = F.ctc_loss(predict_ts_ctc, text, mel_lengths_downsample, text_lengths, blank=0)
                print('loss_ctc = {}'.format(loss_ctc.item()))
                loss = (hp.mlt_weight * loss_att + (1 - hp.mlt_weight) * loss_ctc) / hp.accum_grad
            else:
                loss = loss_att
            print('loss =', loss.item())
        if not torch.isnan(loss):
            if hp.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
                scaler.step(optimizer)
                scaler.update()
                print(f'backward {time.time() - local_time}')
                local_time = time.time()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
                optimizer.step()
        else:
            print('loss is nan')
            sys.exit(1)

        sys.stdout.flush()
        # calc
    n_correct = 0
    for i, t in enumerate(text_lengths):
        tmp = youtputs[i, :t-1, :].max(1)[1].cpu().numpy()
        for j in range(t-1):
            if tmp[j] == text[i][j+1]:
                n_correct = n_correct + 1
    acc = 1.0 * n_correct / float(sum(text_lengths))
    print('acc = {}'.format(acc))
    if rank == 0 and ((epoch+1) % hp.save_per_epoch >= (hp.save_per_epoch - 10) or (epoch+1) % hp.save_per_epoch == 0):
        #torch.save(model.to('cpu').state_dict(), hp.save_dir+"/network.epoch{}".format(epoch+1))
        torch.save(model.state_dict(), hp.save_dir+"/network.epoch{}".format(epoch+1))
        print('save model')
        #model = model.to(rank)
        #torch.save(model.state_dict(), hp.save_dir+"/network.epoch{}".format(epoch+1))
    
    if rank==0 and (epoch+1) % hp.save_per_epoch == 0:
        torch.save(optimizer.state_dict(), hp.save_dir+"/network.optimizer.epoch{}".format(epoch+1))
        print('save optimizer')

    dist.barrier()
    return step

def train_epoch(model, optimizer, args, hp, step, start_epoch=0, rank=0):
    for epoch in range(start_epoch, hp.max_epoch):
        start_time = time.time()

        step = train_loop(model, optimizer, step, epoch, args, hp, rank)
 
        print("EPOCH {} end".format(epoch+1))
        print('elapsed time = {}'.format(time.time() - start_time))


def init_distributed(rank, n_gpus):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."

    torch.cuda.set_device(rank % n_gpus)

    os.environ['MASTER_ADDR'] = 'localhost' #dist_config.MASTER_ADDR
    os.environ['MASTER_PORT'] = '600010' #dist_config.MASTER_PORT

    torch.distributed.init_process_group(
        backend='nccl', world_size=n_gpus, rank=rank
    )


def cleanup():
    torch.distributed.destroy_process_group()


def run_distributed(fn, args, hp):
    try:
        mp.spawn(fn, args=(args, hp), nprocs=args.n_gpus, join=True)
    except:
        cleanup()

def run_training(rank, args, hp):
    if args.n_gpus > 1:
        init_distributed(rank, args.n_gpus)
        torch.cuda.set_device(f'cuda:{rank}')

    model = Transformer(hp)
    model.apply(init_weight)
    model.train()

    model = model.to(rank)
    if args.n_gpus > 1:
        model = DDP(model, device_ids=[rank])
    
    max_lr = hp.init_lr
    if hp.optimizer_type == 'Noam':
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    
    assert (hp.batch_size is None) != (hp.max_seqlen is None)

    
    if hp.loaded_epoch is not None:
        start_epoch = hp.loaded_epoch
        load_dir = hp.loaded_dir
        print('epoch {} loaded'.format(hp.loaded_epoch))
        loaded_dict = load_model("{}".format(os.path.join(load_dir, 'network.epoch{}'.format(hp.loaded_epoch))))
        model.load_state_dict(loaded_dict)
        if hp.is_flat_start:
            step = 1
            start_epoch = 0
            print('flat_start')
        else:
            train_dataset = datasets.get_dataset(hp.train_script, hp.spm_model, hp, spec_aug=False, feat_norm=[hp.mean_file, hp.var_file])
            collate_fn_transformer = datasets.collate_fn
            if hp.batch_size is not None:
                sampler = datasets.NumBatchSampler(train_dataset, hp.batch_size)
            elif hp.max_seqlen is not None:
                sampler = datasets.LengthsBatchSampler(train_dataset, hp.max_seqlen, hp.lengths_file, shuffle=True, shuffle_one_time=False, shuffle_all=hp.dataset_shuffle_all)
            train_sampler = DistributedSampler(sampler) if args.n_gpus > 1 else sampler
            dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, collate_fn=collate_fn_transformer)
            loaded_dict = torch.load("{}".format(os.path.join(load_dir, 'network.optimizer.epoch{}'.format(hp.loaded_epoch))))
            optimizer.load_state_dict(loaded_dict)
            step = hp.loaded_epoch * len(dataloader)
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
        run_training(0, args, hp)
