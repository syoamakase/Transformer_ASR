# -*- coding: utf-8 -*-
import copy
import numpy as np
from struct import unpack
import os
import sys
import sentencepiece as spm
import torch
import torch.nn as nn
from shutil import copyfile

#import hparams as hp
#from utils import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_config(hp):
    print('PID = {}'.format(os.getpid()))
    #print('cuda device = {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    print('PyTorch version =', torch.__version__)
    print('HOST =', os.uname()[1])
    for key in hp.__dict__.keys():
        if not '__' in key:
            print('{} = {}'.format(key, eval('hp.'+key)))

def load_dat(filename):
    """
    To read binary data in htk file.
    The htk file includes log mel-scale filter bank.

    Args:
        filename : file name to read htk file

    Returns:
        dat : 120 (means log mel-scale filter bank) x T (time frame)

    """
    fh = open(filename, "rb")
    spam = fh.read(12)
    _, _, sampSize, _ = unpack(">IIHH", spam)
    veclen = int(sampSize / 4)
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype=np.float32)
    dat = dat.reshape(int(len(dat) / veclen), veclen)
    dat = dat.byteswap()
    fh.close()
    return dat

def onehot(labels, num_output):
    """
    To make onehot vector.
    ex) labels : 3 -> [0, 0, 1, 0, ...]

    Args:
        labels : true label ID
        num_output : the number of entry

    Returns:
        utt_label : one hot vector.
    """
    utt_label = np.zeros((len(labels), num_output), dtype='float32')
    for i in range(len(labels)):
        utt_label[i][labels[i]] = 1.0
    return utt_label


def load_model(model_file, map_location=DEVICE):
    """
    To load PyTorch models either of single-gpu and multi-gpu based model
    """
    model_state = torch.load(model_file, map_location=map_location)
    is_multi_loading = True if torch.cuda.device_count() > 1 else False
    # This line may include bugs!!
    is_multi_loaded = True if 'module' in list(model_state.keys())[0] else False

    if is_multi_loaded is is_multi_loading:
        return model_state

    # the model to load is multi-gpu and the model to use is single-gpu
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

def init_weight(m):
    """ 
    To initialize weights and biases.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)

    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)


def frame_stacking(mel_input, pos_mel, frame_stacking):
    pass

def npeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = (torch.from_numpy(np_mask) == 0).to(DEVICE)
    return np_mask

def create_masks(src_pos, trg_pos, src_pad=0, trg_pad=0):
    src_mask = (src_pos != src_pad).unsqueeze(-2)

    if trg_pos is not None:
        trg_mask = (trg_pos != trg_pad).unsqueeze(-2)
    
        size = trg_pos.size(1) # get seq_len for matrix
        np_mask = npeak_mask(size)
        if trg_pos.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask

def adjust_learning_rate(optimizer, epoch):
    if epoch > 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.8 

def save_hparams(base_file, save_file):
    if os.path.exists(save_file):
        pass
    else:
        copyfile(base_file, save_file)

def overwrite_hparams(args):
    for key, value in args._get_kwargs():
        if value is not None and value != 'load_name':
            setattr(hp, key, value)

def fill_variables(hp, verbose=True):
    default_var = {'pe_alpha': False, 'stop_lr_change': 100000000, 'feed_forward': 'linear', 'optimizer': 'adam', 'mel_dim':80, 'is_flat_start':False,'dataset_shuffle_all':False, 'optimizer_type': 'Noam', 'init_lr':1e-3, 'save_per_epoch': 50, 'save_attention_per_step': 2000, 'num_F':2,
                    'accum_grad':1, 'N_e':12, 'N_d':6, 'heads':4, 'd_model_e':256, 'd_model_d':256, 'encoder': None, 'amp': False, 'comment':'', 'granularity':1, 'subsampling_rate': 4, 'frame_stacking':1, 'decoder_rel_pos':False, 'random_mask':False, 'decoder': 'Transformer'}
    for key, value in default_var.items():
        if not hasattr(hp, key):
            if verbose:
                print('{} is not found in hparams. defalut {} is used.'.format(key, value))
            setattr(hp, key, value)

def decode_ids(spm_model, text_seq):
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model)
    return sp.DecodeIds(text_seq)
