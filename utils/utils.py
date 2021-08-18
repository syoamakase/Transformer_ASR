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
import scipy.io.wavfile

#import hparams as hp
#from utils import hparams as hp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_lmfb_from_wav(hp, load_file):
    def init_fbank():
        ms = mhi - mlo;
        cf = np.zeros((maxChan + 1))
        for chan in range(1, maxChan + 1):
            cf[chan] = (1.0 * chan / maxChan) * ms + mlo

        # create lochan map
        loChan = np.zeros((Nby2 + 1), dtype=np.int)
        chan = 1
        for k in range(1, Nby2 + 1):
            if k < klo or k > khi:
                loChan[k] = -1
            else:
                melk = Mel(k, fres)
                while (cf[chan] < melk) and (chan <= maxChan):
                    chan = chan + 1
                    if not (cf[chan] < melk and chan <= maxChan):
                        break
                loChan[k] = chan - 1

        loWt = np.zeros((Nby2 + 1))
        for k in range(1, Nby2 + 1):
            chan = loChan[k]
            if k < klo or k > khi:
                loWt[k] = 0.0
            else:
                if chan > 0:
                    loWt[k] = ((cf[chan + 1] - Mel(k, fres)) / (cf[chan + 1] - cf[chan]))
                else:
                    loWt[k] = (cf[1] - Mel(k, fres)) / (cf[1] - mlo)

        return cf, loChan, loWt

    def get_lmfb(cf, loChan, loWt, htk_ek):
        mfb = np.zeros((htk_ek.shape[0], numChans + 1))

        for k in range(klo, khi + 1):
            ek = htk_ek[:, k]
            bin = loChan[k]
            t1 = loWt[k] * ek
            if bin > 0:
                mfb[:, bin] += t1
            if bin < numChans:
                mfb[:, bin + 1] += ek - t1

        return np.log(np.clip(mfb[:, 1:numChans+1], 1e-8, None))
    
    def Mel(k, fresh):
        return 1127 * np.log(1 + (k - 1) * fres)

    def get_spec(signal):
        num_frames = int(np.ceil(float(np.abs(len(signal) - frame_length)) / frame_step))
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        original_frames = signal[indices.astype(np.int32, copy=False)]
        # preemp
        frames = np.hstack((original_frames[:, 0:1] * (1.0 - pre_emphasis), original_frames[:, 1:] - pre_emphasis * original_frames[:, :-1]))
        # hamming
        frames *= np.hamming(frame_length)
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        htk_ek = np.hstack((mag_frames[:, 0:1], mag_frames[:, 0:256]))
        return htk_ek

    pre_emphasis = 0.97
    frame_length = 400
    frame_step = 160
    NFFT = 512
    Nby2 = int(NFFT / 2)
    fres = 16000 / (NFFT * 700)
    klo = 2
    khi = Nby2
    mlo = 0.0
    mhi = Mel(Nby2 + 1, fres)
    numChans = hp.mel_dim
    maxChan = numChans + 1

    cf, loChan, loWt = init_fbank()
    sample_rate, signal = scipy.io.wavfile.read(load_file)
    lmfb = get_lmfb(cf, loChan, loWt, get_spec(signal))
    return lmfb

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
        if classname.find('DropLinear') != -1:
            pass
        else:
            m.weight.data.uniform_(-0.1, 0.1)
            if isinstance(m.bias, nn.parameter.Parameter):
                m.bias.data.fill_(0)

    if classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'weight' in name and not 'norm' in name:
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


def average_checkpoints(start, end, hp, multi_gpu=False):
    last = []
    dirname = hp.save_dir
    for epoch in range(start, end+1):
       last.append(os.path.join(dirname, 'network.epoch{}'.format(epoch)))
        
    print("average over", last)
    avg = None

    # sum
    for path in last:
        print(path)
        #states = torch.load(path, map_location=torch.device("cpu"))["model"]
        states = torch.load(path, map_location=torch.device("cpu"))
        if avg is None:
            avg = {}
            for k in states.keys():
                if multi_gpu:
                    avg[k[7:]] = states[k]
                else:
                    avg[k] = states[k]
        else:
            for k in states.keys():
                if multi_gpu:
                    avg[k[7:]] += states[k]
                else:
                    avg[k] = states[k]

    # average
    for k in avg.keys():
        if avg[k] is not None:
            avg[k] = torch.div(avg[k], end-start+1)

    #torch.save(avg, args.out)
    #print('{} saved.'.format(args.out))

    return avg


def frame_stacking(x, pos_x, stack):
    """ frame stacking.

    Args:
        x (Tensor): The input data (basically log mel-scale filter bank features).
        x_lengths (list): The lengths of x.
        stack (int): The factor of frame stacking.
    Returns:
        Tensor: Stacked x. the lengths of x is len(x) // stack
        list: The lengths of stacked x.
    """
    if stack == 1:
        return x, pos_x
    else:
        batch_size = x.shape[0]
        newlen = x.shape[1] // stack
        pos_x = pos_x[:, :newlen]
        stacked_x = x[:, 0:newlen*stack].reshape(batch_size, newlen, -1)
        return stacked_x, pos_x


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
        #if trg_pos.is_cuda:
        np_mask.to(trg_pos.device)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask

def adjust_learning_rate(optimizer, epoch, threshold_epoch=20):
    lr = 0
    if epoch >= threshold_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.8
            lr = param_group['lr']
    return lr

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
                    'accum_grad':1, 'N_e':12, 'N_d':6, 'heads':4, 'd_model_e':256, 'd_model_d':256, 'encoder': None, 'amp': False, 'comment':'', 'granularity':1, 'subsampling_rate': 4, 'frame_stacking':1, 'decoder_rel_pos':False, 'random_mask':False, 'decoder': 'Transformer', 'cnn_avepool':False,
                    'decay_epoch': 100000, 'mean_utt':False, 'multihead':False, 'l1_flag':False, 'load_name_lm': None, 'use_ctc':False, 'lm_weight': None, 'cnn_swish':False, 'cnn_ln': False, 'beam_width':10}
    for key, value in default_var.items():
        if not hasattr(hp, key):
            if verbose:
                print('{} is not found in hparams. defalut {} is used.'.format(key, value))
            setattr(hp, key, value)

    dev_var = {'swish_lstm':False, 'norm_lstm':False, 'load_name_lm_2':None, 'weight_dropout':None, 'dev_mode': None}
    for key, value in dev_var.items():
        if not hasattr(hp, key):
            if verbose:
                print('{} is not found in hparams in development. defalut {} is used.'.format(key, value))
            setattr(hp, key, value)

def decode_ids(spm_model, text_seq):
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model)
    return sp.DecodeIds(text_seq)
