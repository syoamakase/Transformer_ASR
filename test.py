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
from utils.utils import fill_variables, load_dat, load_lmfb_from_wav
import utils
import time
from Models.transformer import Transformer
from Models.LM import Model_lm, TransformerLM

from tools.calc_wer import wer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            #if 'cnn_encoder.conv.2' in key:
            #    new_key = key.replace('cnn_encoder.conv.2', 'cnn_encoder.conv.1')
            #    new_model_state[new_key[7:]] = model_state[key]       
            #elif 'cnn_encoder.out.0' in key:
            #    new_key = key.replace('cnn_encoder.out.0', 'cnn_encoder.out')
            #    new_model_state[new_key[7:]] = model_state[key]       
            #else:
            #    new_model_state[key[7:]] = model_state[key]
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

def recognize(hp, model, script_file, model_lm, lm_weight, model_lm_2=None, lm_weight_2=None, calc_wer=False):
    # TODO: calculate wer
    sp = spm.SentencePieceProcessor()
    sp.Load(hp.spm_model)
    INIT_TOK = sp.bos_id()
    EOS_TOK = sp.eos_id()
    BATCH_SIZE = 1 
    script_buf = []
    with open(script_file) as f:
        for line in f:
            script_buf.append(line)
    
    num_mb = len(script_buf) // BATCH_SIZE
    results_all = np.zeros(5)
    sil = np.load(hp.silence_file) if hp.silence_file is not None else None
    start_time = time.time()
    for i in range(num_mb):
        xs = []
        lengths = []
        ts_lengths = []
        for j in range(BATCH_SIZE):
            s = script_buf[i*BATCH_SIZE+j].strip()
            if len(s.split('|')) == 1:
                x_file = s
            else:
                x_file, laborg = s.split('|', 1)
            if '.htk' in x_file:
                cpudat = load_dat(x_file)
                cpudat = cpudat[:, :hp.mel_dim]

            elif '.wav' in x_file:
                cpudat = load_lmfb_from_wav(hp, x_file)
            elif '.npy' in x_file:
                cpudat = np.load(x_file)
            if sil is not None:
                cpudat = np.vstack((cpudat, sil))
            if hp.mean_utt:
                cpudat = cpudat - cpudat.mean(axis=0, keepdims=True)
            if hp.mean_file is not None and hp.var_file is not None:
                mean = np.load(hp.mean_file).reshape(1, -1)
                var = np.load(hp.var_file).reshape(1, -1)
                cpudat = (cpudat - mean) / np.sqrt(var)

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
        result_print = ''
        if src_seq.shape[1] <= 10:
            continue
        if not calc_wer:
            result_print = f"{x_file.strip()} "
        if src_seq.shape[1] >= args.segment:
            seg = args.segment - 100
            for i in range(1, src_seq.shape[1] // (args.segment-100) + 2):
                youtput_in_Variable = model.decode(src_seq[:, (i-1)*seg:i*seg], src_seq_dummy[:, (i-1)*seg:i*seg], hp.beam_width, model_lm, INIT_TOK, EOS_TOK, lm_weight)
                if len(youtput_in_Variable) != 0:
                    result_print += f"{sp.DecodeIds(youtput_in_Variable)} "
        else:
            youtput_in_Variable = model.decode(src_seq, src_seq_dummy, hp.beam_width, model_lm, INIT_TOK, EOS_TOK, lm_weight, model_lm_2, lm_weight_2)
            if len(youtput_in_Variable) != 0:
                result_print += f"{sp.DecodeIds(youtput_in_Variable)}"
            # segment the speech in evaluation
            elif src_seq.shape[1] > 1000:
                seg = 1000 - 100
                for i in range(1, src_seq.shape[1] // (args.segment-100) + 2):
                    youtput_in_Variable = model.decode(src_seq[:, (i-1)*seg:i*seg], src_seq_dummy[:, (i-1)*seg:i*seg], hp.beam_width, model_lm, INIT_TOK, EOS_TOK, lm_weight)
                    if len(youtput_in_Variable) != 0:
                        result_print += f"{sp.DecodeIds(youtput_in_Variable)} "
                

        if not calc_wer:
            print(result_print)
            sys.stdout.flush()
        else:
            results_all += wer(laborg.split(), result_print.split())

    print(f'elapsed time = {time.time() -start_time}')
    if calc_wer:
        wer_results_all = results_all[1:-1].sum()/ results_all[-1]
        results_all = results_all.astype(np.int32)
        print('WER {0:.2f}% [H={1:d}, D={2:d}, S={3:d}, I={4:d}, N={5:d}]'.format(results_all[1:-1].sum()/ results_all[-1] * 100, results_all[0], results_all[1], results_all[2], results_all[3], results_all[4]))
    
        #print('WER is ', wer_results_all)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_name')
    parser.add_argument('--hp_file', type=str, default=None)
    parser.add_argument('--test_script', type=str, default=None)
    parser.add_argument('--load_name_lm', type=str, default=None)
    parser.add_argument('--lm_weight', type=float, default=None)
    parser.add_argument('--load_name_lm_2', type=str, default=None)
    parser.add_argument('--lm_weight_2', type=float, default=0.2)
    parser.add_argument('--beam_width', type=int, default=None)
    parser.add_argument('--log_params', action='store_true')
    parser.add_argument('--calc_wer', action='store_true')
    parser.add_argument('--segment', type=int, default=10000)
    parser.add_argument('--silence_file', type=str, default=None)
    parser.add_argument('--lm_type', type=str, default='LSTM')
    args = parser.parse_args()
    hp_file = args.hp_file
    model_name = args.load_name # save dir name
    
    model_path = os.path.dirname(model_name)
    
    if hp_file is None:
        hp_file = os.path.join(model_path, 'hparams.py')
    
    hp.configure(hp_file)
    fill_variables(hp)
    
    setattr(hp, 'silence_file', args.silence_file)

    if args.beam_width is not None:
        print(f'beam width is set to {args.beam_width}')
        hp.beam_width = args.beam_width

    script_file = hp.eval_file
    if args.test_script is not None:
        script_file = args.test_script

    if hp.lm_weight is not None:
        if args.lm_weight is None:
            args.lm_weight = hp.lm_weight
        else:
            print(f'lm_weight {args.lm_weight} on args is used')        

    print(f'lm weight = {args.lm_weight}')
    model = Transformer(hp)
    model.to(DEVICE)
    model.eval()

    assert (args.load_name_lm is None and hp.load_name_lm is not None) or (args.load_name_lm is not None and hp.load_name_lm is None) or (args.load_name_lm is None and hp.load_name_lm is None) , \
            f'You specified load_name_lm on a command line argument and hparams. Which do you want to use?'

    if args.load_name_lm is not None:
        print(f'load {args.load_name_lm}')
        hp_LM_path = os.path.join(os.path.dirname(args.load_name_lm), 'hparams.py')
        hp_LM = utils.HParams()
        hp_LM.configure(hp_LM_path)
        if args.lm_type == 'LSTM':
            model_lm = Model_lm(hp_LM)
        else:
            model_lm = TransformerLM(hp_LM)
        model_lm.to(DEVICE)
        model_lm.load_state_dict(load_model(args.load_name_lm))
        model_lm.eval()
    else:
        model_lm = None

    if hp.load_name_lm is not None:
        print(f'load {hp.load_name_lm}')
        hp_LM_path = os.path.join(os.path.dirname(hp.load_name_lm), 'hparams.py')
        hp_LM = utils.HParams()
        hp_LM.configure(hp_LM_path)
        if args.lm_type == 'LSTM':
            model_lm = Model_lm(hp_LM)
        else:
            model_lm = TransformerLM(hp_LM)
        model_lm.to(DEVICE)
        model_lm.load_state_dict(load_model(hp.load_name_lm))
        model_lm.eval()


    if args.load_name_lm_2 is not None:
        print(f'load {args.load_name_lm_2}')
        hp_LM_path = os.path.join(os.path.dirname(args.load_name_lm_2), 'hparams.py')
        hp_LM = utils.HParams()
        hp_LM.configure(hp_LM_path)
        model_lm_2 = Model_lm(hp_LM)
        model_lm_2.to(DEVICE)
        model_lm_2.load_state_dict(load_model(args.load_name_lm_2))
        model_lm_2.eval()
    else:
        model_lm_2 = None

    if hp.load_name_lm_2 is not None:
        print(f'load {hp.load_name_lm}')
        hp_LM_path = os.path.join(os.path.dirname(hp.load_name_lm_2), 'hparams.py')
        hp_LM = utils.HParams()
        hp_LM.configure(hp_LM_path)
        model_lm_2 = Model_lm(hp_LM)
        model_lm_2.to(DEVICE)
        model_lm_2.load_state_dict(load_model(hp.load_name_lm_2))
        model_lm_2.eval()

    model.load_state_dict(load_model(model_name))
    
    recognize(hp, model, script_file, model_lm, args.lm_weight, model_lm_2, args.lm_weight_2, calc_wer=args.calc_wer)
