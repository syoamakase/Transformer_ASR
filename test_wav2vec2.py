# -*- coding: utf-8 -*-
"""
decoding
"""
import argparse
import math
import numpy as np
import os
import sentencepiece as spm
import soundfile as sf
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from utils import hparams as hp
from utils.utils import fill_variables, load_dat
import utils
#from Models.transformer import Transformer
from Models.transformer_wav2vec2 import TransformerWav2vec2
from Models.LM import Model_lm
from transformers import Wav2Vec2Processor

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

def recognize(hp, model, script_file, model_lm, lm_weight, processor, calc_wer=False):
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
            if '.wav' in x_file:
                audio_input, sampling_rate = sf.read(x_file)
                wav_input = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_values
                wav2vec2_length = math.floor((wav_input.shape[1] - 400) / 320.) + 1
    
            lengths.append(wav2vec2_length)
            xs.append(wav_input)

        ## TODO: change preprocess
        xs_dummy = []
        src_pad = 0
        for i in range(len(xs)):
            xs_dummy.append([1] * lengths[i])
        #src_seq = np.zeros((BATCH_SIZE, max(lengths), hp.mel_dim))
        #for i in range(len(xs)):
        #    src_seq[i, :lengths[i], :] = xs[i]
        src_seq_dummy = np.array([inst + [src_pad] * (max(lengths) - len(inst)) for inst in xs_dummy])
    
        src_seq = torch.tensor(xs[0]).to(DEVICE).float()
        src_seq_dummy = torch.from_numpy(src_seq_dummy).to(DEVICE).long()
        ## TODO: adjust wav lengths
        if False: #src_seq.shape[1] >= 2000:
            print("{}".format(x_file.strip()), end =' ')
            for i in range(1, src_seq.shape[1] // 1900 + 2):
                youtput_in_Variable = model.decode(src_seq[:, (i-1)*1900:i*1900], src_seq_dummy[:, (i-1)*1900:i*1900], 10, model_lm, INIT_TOK, EOS_TOK, lm_weight)
                if len(youtput_in_Variable) != 0:
                    print("{}".format(sp.DecodeIds(youtput_in_Variable)), end=' ')
            print()
        else:
            youtput_in_Variable = model.decode(src_seq, src_seq_dummy, 10, model_lm, INIT_TOK, EOS_TOK, lm_weight)
            if len(youtput_in_Variable) == 0:
                print("{}".format(x_file.strip()))
            else:
                print("{} {}".format(x_file.strip(), sp.DecodeIds(youtput_in_Variable)))
        sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_name')
    parser.add_argument('--hp_file', type=str, default=None)
    parser.add_argument('--test_script', type=str, default=None)
    parser.add_argument('--load_name_lm', type=str, default=None)
    parser.add_argument('--lm_weight', type=float, default=0.2)
    parser.add_argument('--log_params', action='store_true')
    args = parser.parse_args()
    hp_file = args.hp_file
    model_name = args.load_name # save dir name
    
    model_path = os.path.dirname(model_name)
    
    if hp_file is None:
        hp_file = os.path.join(model_path, 'hparams.py')
    
    hp.configure(hp_file)
    fill_variables(hp)
    
    script_file = hp.eval_file
    if args.test_script is not None:
        script_file = args.test_script

    model = TransformerWav2vec2(hp, pretrain_model='facebook/wav2vec2-large-lv60')
    model.to(DEVICE)
    model.eval()

    assert (args.load_name_lm is None and hp.load_name_lm is not None) or (args.load_name_lm is not None and hp.load_name_lm is None) or (args.load_name_lm is None and hp.load_name_lm is None) , \
            f'You specified load_name_lm on a command line argument and hparams. Which do you want to use?'

    if args.load_name_lm is not None:
        print(f'load {args.load_name_lm}')
        hp_LM_path = os.path.join(os.path.dirname(args.load_name_lm), 'hparams.py')
        hp_LM = utils.HParams()
        hp_LM.configure(hp_LM_path)
        model_lm = Model_lm(hp_LM)
        model_lm.to(DEVICE)
        model_lm.load_state_dict(load_model(args.load_name_lm))
        model_lm.eval()
    else:
        model_lm = None

    if hp.load_name_lm is not None:
        print(f'load {hp.load_name_lm} ')
        hp_LM_path = os.path.join(os.path.dirname(hp.load_name_lm), 'hparams.py')
        hp_LM = utils.HParams()
        hp_LM.configure(hp_LM_path)
        model_lm = Model_lm(hp_LM)
        model_lm.to(DEVICE)
        model_lm.load_state_dict(load_model(hp.load_name_lm))
        model_lm.eval()

    model.load_state_dict(load_model(model_name))
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-lv60")

    recognize(hp, model, script_file, model_lm, args.lm_weight, processor=processor)
