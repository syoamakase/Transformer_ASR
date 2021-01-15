# -*- coding: utf-8 -*-
"""
decoding
"""
import argparse
import os
import sys

import numpy as np
import math

import sentencepiece as spm
import torch
import torch.nn.functional as F
import torch.nn as nn

from utils import hparams as hp
from utils.utils import fill_variables
from Models.transformer import Transformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dat(filename):
    fh = open(filename, "rb")
    spam = fh.read(12)
    nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
    veclen = int(sampSize / 4)
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype=np.float32)
    dat = dat.reshape(int(len(dat) / veclen), veclen)
    dat = dat.byteswap()
    fh.close()
    return dat

def load_model(model_file):
    model_state = torch.load(model_file)
    is_multi_loading = True if torch.cuda.device_count() > 1 else False
    is_multi_loaded = True if 'module' in list(model_state.keys())[0] else False

    if is_multi_loaded is is_multi_loading:
        return model_state

    elif is_multi_loaded is False and is_multi_loading is True:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state['module.'+key] = model_state[key]

        return new_model_state

    elif is_multi_loaded is True and is_multi_loading is False:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state[key[7:]] = model_state[key]

        return new_model_state

    else:
        print('ERROR in load model')
        sys.exit(1)

def npeak_mask(size):
    """
    npeak_mask(4)
    >> tensor([[[ 1,  0,  0,  0],
         [ 1,  1,  0,  0],
         [ 1,  1,  1,  0],
         [ 1,  1,  1,  1]]], dtype=torch.uint8)

    """
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = (torch.from_numpy(np_mask) == 0).to(DEVICE)
    return np_mask

def create_masks(src, trg, src_pad, trg_pad):
    src_mask = (src != src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = npeak_mask(size)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_name')
    parser.add_argument('--hp_file', type=str, default=None)
    args = parser.parse_args()
    hp_file = args.hp_file
    model_name = args.load_name # save dir name
    
    model_path  = os.path.dirname(model_name)
    
    if hp_file is None:
        hp_file = os.path.join(model_path, 'hparams.py')
    
    hp.configure(hp_file)
    fill_variables()
    
    sp = spm.SentencePieceProcessor()
    sp.Load(hp.spm_model)
    
    FRAME_STACKING = 1
    INIT_TOK = sp.bos_id()
    EOS_TOK = sp.eos_id()
    NUM_CLASSES = sp.get_piece_size()
    BATCH_SIZE = 1 
    script_file = hp.eval_file

    model = Transformer(hp)
    model.to(DEVICE)
    model.eval()
    
    # load
    model.load_state_dict(load_model(model_name))
    
    script_buf = []
    with open(script_file) as f:
        for line in f:
            script_buf.append(line)
    
    num_mb = len(script_buf) // BATCH_SIZE
    for i in range(num_mb):
        xs = []
        lengths = []
        ts_lengths = []
        for j in range(BATCH_SIZE):
            s = script_buf[i*BATCH_SIZE+j].strip()
            if len(s.split(' ')) == 1:
                x_file = s
            else:
                x_file, laborg = s.split(' ', 1)
            if '.htk' in x_file:
                cpudat = load_dat(x_file)
                cpudat = cpudat[:, :hp.mel_dim]
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
    
        src_seq = torch.from_numpy(src_seq).to(DEVICE).float()
        src_seq_dummy = torch.from_numpy(src_seq_dummy).to(DEVICE).long()
        youtput_in_Variable = model.decode(src_seq, src_seq_dummy, 10, None, INIT_TOK, EOS_TOK)
        #Beam_cnn_v2.beam_search(src_seq, src_seq_dummy, model, INIT_TOK, EOS_TOK)
        if len(youtput_in_Variable) == 0:
            print("{}".format(x_file.strip()))
        else:
            print("{} {}".format(x_file.strip(), sp.DecodeIds(youtput_in_Variable)))
        sys.stdout.flush()
