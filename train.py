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
from torch.utils.data import DataLoader

from utils import hparams as hp
from utils.utils import log_config, fill_variables, adjust_learning_rate, load_model, create_masks, init_weight
from Models.transformer import Transformer

random.seed(77)
torch.random.manual_seed(777)
torch.cuda.manual_seed_all(777)
np.random.seed(777)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_learning_rate(step):
    d_model = 256
    return warmup_factor * min(step ** -0.5, step * warmup_step ** -1.5) * (d_model ** -0.5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', type=str, default='hparams.py')
    args = parser.parse_args()
    hp_file = args.hp_file
    hp.configure(hp_file)
    fill_variables()
    log_config()

    save_dir = hp.save_dir
    label_smoothing = True
    clip = 5.0

    model = Transformer(hp)

    # multi-gpu setup
    if torch.cuda.device_count() > 1:
        # multi-gpu configuration
        ngpu = torch.cuda.device_count()
        device_ids = list(range(ngpu))
        model = torch.nn.DataParallel(model, device_ids)
        model.cuda()
    else:
        model.to(DEVICE)
    
    model.apply(init_weight)
    model.train()
    
    max_lr = hp.init_lr 
    warmup_step = hp.warmup_step # 4000
    warmup_factor = hp.warmup_factor #10.0 # 1.0
    # if hp.optimizer.lower() == 'adam': 
    #     optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
    # elif hp.optimizer.lower() == 'radam':
    #     import radam
    #     optimizer = radam.RAdam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
    if hp.optimizer_type == 'Noam':
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    
    os.makedirs(save_dir, exist_ok=True)
    #print('Weight decay = ', optimizer.param_groups[0]['weight_decay'])
    
    assert (hp.batch_size is None) != (hp.max_seqlen is None)
    
    if hp.loaded_epoch is not None:
        start_epoch = hp.loaded_epoch
        load_dir = hp.loaded_dir
        print('epoch {} loaded'.format(hp.loaded_epoch))
        model.load_state_dict(load_model("{}".format(os.path.join(load_dir, 'network.epoch{}'.format(hp.loaded_epoch)))))
        if hp.is_flat_start:
            step = 1
            start_epoch = 0
            print('flat_start')
        else:
            train_dataset = datasets.get_dataset(hp.train_script, hp.spm_model, spec_aug=False, feat_norm=[hp.mean_file, hp.var_file])
            collate_fn_transformer = datasets.collate_fn
            if hp.batch_size is not None:
                sampler = datasets.NumBatchSampler(train_dataset, hp.batch_size)
            elif hp.max_seqlen is not None:
                sampler = datasets.LengthsBatchSampler(train_dataset, hp.max_seqlen, hp.lengths_file, shuffle=True, shuffle_one_time=False, shuffle_all=hp.dataset_shuffle_all)
            dataloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4, collate_fn=collate_fn_transformer)
            optimizer.load_state_dict(torch.load("{}".format(os.path.join(load_dir, 'network.optimizer.epoch{}'.format(hp.loaded_epoch)))))
            step = hp.loaded_epoch * len(dataloader)
    else:
        start_epoch = 0
        step = 1
    
    scaler = torch.cuda.amp.GradScaler()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('params = {0:.2f}M'.format(pytorch_total_params / 1000 / 1000))
    
    for epoch in range(start_epoch, hp.max_epoch):
        src_pad = 0
        trg_pad = 0
        if step > hp.warmup_step:
            train_dataset = datasets.get_dataset(hp.train_script, hp.spm_model, spec_aug=hp.use_spec_aug, feat_norm=[hp.mean_file, hp.var_file])
        else:
            train_dataset = datasets.get_dataset(hp.train_script, hp.spm_model, spec_aug=False, feat_norm=[hp.mean_file, hp.var_file])

        collate_fn_transformer = datasets.collate_fn
        if hp.batch_size is not None:
            sampler = datasets.NumBatchSampler(train_dataset, hp.batch_size)
        elif hp.max_seqlen is not None:
            sampler = datasets.LengthsBatchSampler(train_dataset, hp.max_seqlen, hp.lengths_file, shuffle=True, shuffle_one_time=False, shuffle_all=hp.dataset_shuffle_all)

        dataloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4, collate_fn=collate_fn_transformer)

        #scheduler.step(epoch)
        #pbar = tqdm(dataloader)
        #for d in pbar:
        train_len = len(dataloader)
        start_time = time.time()

        for d in dataloader:
            if hp.optimizer_type == 'Noam':
                lr = get_learning_rate(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                adjust_learning_rate(optimizer, epoch+1)
 
            text, mel_input, pos_text, pos_mel, text_lengths, mel_lengths = d
 
            text = text.to(DEVICE, non_blocking=True)
            mel_input = mel_input.to(DEVICE, non_blocking=True)
            pos_text = pos_text.to(DEVICE, non_blocking=True)
            pos_mel = pos_mel.to(DEVICE, non_blocking=True)
            text_lengths = text_lengths.to(DEVICE, non_blocking=True)
    
            batch_size = mel_input.shape[0]
    
            text_input = text[:, :-1]
            src_mask, trg_mask = create_masks(pos_mel, pos_text[:, :-1])
    
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(hp.amp): #and torch.autograd.set_detect_anomaly(True):
                if hp.mode == 'ctc-transformer':
                    youtputs, ctc_outputs, attn_enc_enc, attn_dec_dec, attn_dec_enc = model(mel_input, text_input, src_mask, trg_mask)
                else:
                    youtputs = model(mel_input, text_input, src_mask, trg_mask)
    
                loss_att = 0.0
                # cross entropy
                if label_smoothing:
                    ys = text[:,1:].contiguous().view(-1)
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
                if torch.isnan(loss):
                    print('loss is nan')
                    sys.exit()
    
                sys.stdout.flush()
                
            if hp.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
                optimizer.step()

        # calc
        n_correct = 0
        for i, t in enumerate(text_lengths):
            tmp = youtputs[i, :t-1, :].max(1)[1].cpu().numpy()
            for j in range(t-1):
                if tmp[j] == text[i][j+1]:
                    n_correct = n_correct + 1
        acc = 1.0 * n_correct / float(sum(text_lengths))
        print('acc = {}'.format(acc))
        if (epoch+1) % hp.save_per_epoch >= (hp.save_per_epoch - 10) or (epoch+1) % hp.save_per_epoch == 0:
            torch.save(model.state_dict(), save_dir+"/network.epoch{}".format(epoch+1))
        #elif (epoch+1) % hp.save_per_epoch > 10:
        #    torch.save(model.state_dict(), save_dir+"/network.epoch{}".format(epoch+1))
           # torch.save(optimizer.state_dict(), save_dir+"/network.optimizer.epoch{}".format(epoch+1))
 
        if (epoch+1) % hp.save_per_epoch == 0:
            torch.save(optimizer.state_dict(), save_dir+"/network.optimizer.epoch{}".format(epoch+1))
 
        print("EPOCH {} end".format(epoch+1))
        print('elapsed time = {}'.format(time.time() - start_time))
