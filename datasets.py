# -*- coding: utf-8 -*-
import argparse
import math
import numpy as np
import pandas as pd
#import librosa
from operator import itemgetter
from typing import Optional
import collections
import os
import random
from struct import unpack, pack
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
import sentencepiece as spm
from tqdm import tqdm


class TrainDatasets(Dataset):
    """
    Dataset class.
    """
    def __init__(self, csv_file, hp, root_dir=None, spec_aug=False, feat_norm=[None, None]):
        """
        Args:                                                                   
            csv_file (string): Path to the csv file with annotations.           
            root_dir (string): Directory with all the wavs.                     
                                                                                
        """
        # self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
          
        self.landmarks_frame = pd.read_csv(csv_file, sep='\|', header=None)
        self.hp = hp
        self.root_dir = root_dir
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.hp.spm_model)
        self.spec_aug = spec_aug
        if feat_norm[0] is not None:
            self.feat_norm = True
            self.mean = np.load(feat_norm[0]).reshape(80)
            self.var = np.load(feat_norm[1]).reshape(80)
        else:
            self.feat_norm = False

        if self.hp.lengths_file is None or not os.path.exists(self.hp.lengths_file):
            print('lengths_file is not exists. Make...')
            lengths_list = []
            pbar = tqdm(range(len(self.landmarks_frame)))
            for idx in pbar:
                mel_name = self.landmarks_frame.loc[idx, 0]
                if '.htk' in mel_name:
                    mel_input = self.load_htk(mel_name)
                    mel_length = mel_input.shape[0]
                elif '.npy' in mel_name:
                    mel_input = np.load(mel_name)
                    mel_length = mel_input.shape[0]
                #mel_length = self.landmarks_frame.loc[idx, 2]
                length = mel_length
                lengths_list.append(length)
                
            self.lengths_np = np.array(lengths_list)
            np.save(self.hp.lengths_file, self.lengths_np)

    def load_htk(self, filename):
        fh = open(filename, "rb")
        spam = fh.read(12)
        _, _, sampSize, _ = unpack(">IIHH", spam)
        veclen = int(sampSize / 4)
        fh.seek(12, 0)
        dat = np.fromfile(fh, dtype='float32')
        dat = dat.reshape(int(len(dat) / veclen), veclen)
        dat = dat.byteswap()
        fh.close()
        return dat                       

    def _freq_mask(self, spec, F=10, num_masks=1, replace_with_zero=False, random_mask=False, granularity=1):
        cloned = spec.clone()
        num_mel_channels = cloned.shape[1]
        for i in range(0, num_masks):
            f = random.randrange(0, F, granularity)
            f_zero = random.randrange(0, num_mel_channels - f)
            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f*granularity): return cloned
            mask_end = random.randrange(f_zero, f_zero + f, granularity)
            if (replace_with_zero): cloned[:, f_zero:mask_end] = 0
            else: cloned[:, f_zero:mask_end] = cloned.mean()
        return cloned
    
    def _time_mask(self, spec, T=50, num_masks=1, replace_with_zero=False, random_mask=False):
        cloned = spec.clone()
        len_spectro = cloned.shape[0]

        for i in range(0, num_masks):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)
    
            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned
    
            mask_end = random.randrange(t_zero, t_zero + t)
            if (replace_with_zero): cloned[t_zero:mask_end,:] = 0
            else: cloned[t_zero:mask_end,:] = cloned.mean()
        return cloned
                                                                                
    def __len__(self):                                                          
        return len(self.landmarks_frame)                                        
                                                                                
    def __getitem__(self, idx): 
        mel_name = self.landmarks_frame.loc[idx, 0]
        text_raw = self.landmarks_frame.loc[idx, 1].strip()
                                                                                
        # text = np.asarray(text_to_sequence(text, [self.hp.cleaners]), dtype=np.int32)
        textids = [self.sp.bos_id()] + self.sp.EncodeAsIds(text_raw) + [self.sp.eos_id()]
        text = np.array([int(t) for t in textids], dtype=np.int32)
        if '.htk' in mel_name:
            mel_input = self.load_htk(mel_name)
            mel_length = mel_input.shape[0]
        elif '.npy' in mel_name:
            mel_input = np.load(mel_name)
            mel_length = mel_input.shape[0]

        if self.feat_norm:
            mel_input = (mel_input - self.mean) / np.sqrt(self.var)

        if self.spec_aug:
            mel_input = torch.from_numpy(mel_input)
            num_T = min(20, math.floor(0.04*mel_length))
            T = math.floor(0.04*mel_length)
            #T = min(mel_input.shape[0] // 2 - 1, 100)
            #mel_input = time_warp(self._time_mask(self._freq_mask(mel_input, F=self.hp.spec_size_f, num_masks=2, replace_with_zero=True), T=T, num_masks=2,replace_with_zero=True))
            #mel_input = self._time_mask(self._freq_mask(mel_input, F=self.hp.spec_size_f, num_masks=2, replace_with_zero=True), T=T, num_masks=2, replace_with_zero=True)
            mel_input = self._time_mask(self._freq_mask(mel_input, F=self.hp.spec_size_f, num_masks=2, replace_with_zero=True, granularity=self.hp.granularity), T=T, num_masks=num_T, replace_with_zero=True)
            mel_length = mel_input.shape[0]
        # mel_input = np.concatenate([np.zeros([1,self.hp.num_mels], np.float32), mel[:-1,:]], axis=0)
        text_length = len(text)                                                 
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, mel_input.shape[0] + 1) 
        #assert (mel_length-2)//4 > text_length, f'mel_length of {mel_name}={(mel_length-2)//4} is shorter than text_length={text_length}'
        if (mel_length-2)//4 <= text_length:
            print(mel_name)
                                                                                
        sample = {'text': text, 'text_length':text_length, 'mel_input':mel_input, 'mel_length':mel_length, 'pos_mel':pos_mel, 'pos_text':pos_text}
                                                                                
        return sample

# class TestDatasets(Dataset):
#     """
#     Test dataset.

#     """                                                   
#     def __init__(self, csv_file, root_dir=None):
#         """                                                                     
#         Args:                                                                   
#             csv_file (string): Path to the csv file with annotations.           
#             root_dir (string): Directory with all the wavs.                     
                                                                                
#         """                                                                     
#         # self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)  
#         self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
#         self.root_dir = root_dir 
                                                                                
#     def load_wav(self, filename):                                               
#         return librosa.load(filename, sr=self.hp.sample_rate) 

#     def load_htk(self, filename):
#         fh = open(filename, "rb")
#         spam = fh.read(12)
#         _, _, sampSize, _ = unpack(">IIHH", spam)
#         veclen = int(sampSize / 4)
#         fh.seek(12, 0)
#         dat = np.fromfile(fh, dtype='float32')
#         dat = dat.reshape(int(len(dat) / veclen), veclen)
#         dat = dat.byteswap()
#         fh.close()
#         return dat 
                                                                                
#     def __len__(self):                                                          
#         return len(self.landmarks_frame)                                        
                                                                                
#     def __getitem__(self, idx): 
#         mel_name = self.landmarks_frame.loc[idx, 0] #os.path.join(self.root_dir, self.landmarks_frame.loc[idx, 0])
                                                                                
#         # text = np.asarray(text_to_sequence(text, [self.hp.cleaners]), dtype=np.int32)
#         text = np.array([int(t) for t in text.split(' ')], dtype=np.int32)
#         if '.htk' in mel_name:
#             mel_input = self.load_htk(mel_name)[:,:40]
#         elif '.npy' in mel_name:
#             mel_input = np.load(mel_name)[:,:40]
#         pos_mel = np.arange(1, mel_input.shape[0] + 1)
                                                                                
#         sample = {'mel_input':mel_input, 'pos_mel':pos_mel, 'mel_name':mel_name}
                                                                                
#         return sample

def collate_fn(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.abc.Mapping):
        text = [d['text'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        mel_lengths = [d['mel_length'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text = [d['pos_text'] for d in batch]
        
        #text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        #mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        #pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        #pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        #text_length = sorted(text_length, reverse=True)
        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)

        return torch.LongTensor(text), torch.FloatTensor(mel_input), torch.LongTensor(pos_text), torch.LongTensor(pos_mel), torch.LongTensor(text_length), torch.LongTensor(mel_lengths)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, self.self.hp.outputs_per_step - (timesteps % self.hp.outputs_per_step)]], mode='constant', constant_values=0.0)

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

class LengthsBatchSampler(Sampler):
    """
    LengthsBatchSampler - Sampler for variable batch size. Mainly, we use it for Transformer.
    It requires lengths.
    """
    def __init__(self, dataset, n_lengths, lengths_file=None, shuffle=True, shuffle_one_time=False, reverse=False, shuffle_all=False):
        assert not ((shuffle == reverse) and shuffle is True), 'shuffle and reverse cannot set True at the same time.'

        print('{} is loading.'.format(lengths_file))
        self.lengths_np = np.load(lengths_file)
        assert len(dataset) == len(self.lengths_np), 'mismatch the number of lines between dataset and {}'.format(lengths_file)
        
        self.n_lengths = n_lengths
        self.shuffle = shuffle
        self.shuffle_one_time = shuffle_one_time
        self.shuffle_all = shuffle_all
        self.reverse = reverse
        self.all_indices = self._batch_indices()
        if shuffle_one_time:
            np.random.shuffle(self.all_indices)

    def _batch_indices(self):
        self.count = 0
        all_indices = []
        
        if not self.shuffle_all:
            while self.count + 1 < len(self.lengths_np):
                indices = []
                max_len = 0
                while self.count < len(self.lengths_np):
                    curr_len = self.lengths_np[self.count]
                    mel_lengths = max(max_len, curr_len) * (len(indices) + 1)
                    if mel_lengths > self.n_lengths or (self.count + 1) > len(self.lengths_np):
                        break
                    #mel_lengths += curr_len
                    max_len = max(max_len, curr_len)
                    indices.extend([self.count])
                    self.count += 1
                all_indices.append(indices)
        else:
            rand_range = np.arange(0, len(self.lengths_np))
            np.random.shuffle(rand_range)
            self.count = 0
            while self.count + 1 < len(self.lengths_np):
                indices = []
                max_len = 0
                while self.count < len(self.lengths_np):
                    idx_rand = rand_range[self.count] 
                    curr_len = self.lengths_np[self.count]
                    mel_lengths = max(max_len, curr_len) * (len(indices) + 1)
                    if mel_lengths > self.n_lengths or (self.count + 1) > len(self.lengths_np):
                        break
                    # mel_lengths += curr_len
                    max_len = max(max_len, curr_len)
                    indices.extend([idx_rand])
                    self.count += 1
                all_indices.append(indices)
       
        return all_indices

    def __iter__(self):
        if self.shuffle and not self.shuffle_one_time:
            np.random.shuffle(self.all_indices)
        if self.reverse:
            self.all_indices.reverse()

        for indices in self.all_indices:
            yield indices
        
        if self.shuffle_all:
            print('shuffle_all')
            self.all_indices = self._batch_indices()

    def __len__(self):
        return len(self.all_indices)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class NumBatchSampler(Sampler):
    """
    LengthsBatchSampler - Sampler for variable batch size. Mainly, we use it for Transformer.
    It requires lengths.
    """
    def __init__(self, dataset, batch_size, drop_last=True):
        self.batch_size = batch_size
        self.drop_last = drop_last 
        self.dataset_len = len(dataset)
        self.all_indices = self._batch_indices()
        np.random.shuffle(self.all_indices)

    def _batch_indices(self):
        batch_len = self.dataset_len // self.batch_size
        mod = self.dataset_len % self.batch_size
        all_indices = np.arange(self.dataset_len-mod).reshape(batch_len,self.batch_size)
       
        return all_indices

    def __iter__(self):
        np.random.shuffle(self.all_indices)

        for indices in self.all_indices:
            yield indices

    def __len__(self):
        return len(self.all_indices)

def get_dataset(script_file, hp, spec_aug=True, feat_norm=[None, None]):
    print('script_file = {}'.format(script_file))
    print('spec_auc = {}'.format(spec_aug))
    print('feat norm = {}'.format(feat_norm))
    return TrainDatasets(script_file, hp, spec_aug=spec_aug, feat_norm=feat_norm)

if __name__ == '__main__':
    from utils import hparams as hp
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py')
    parser.add_argument('--train_script', default=None)
    args = parser.parse_args()

    hp.configure(args.hp_file)
    if args.train_script is not None:
        hp.train_script = args.train_script
    print(f'train script = {hp.train_script}')
    datasets = TrainDatasets(hp.train_script, hp)
    sampler = LengthsBatchSampler(datasets, hp.max_seqlen, hp.lengths_file, shuffle=True, shuffle_one_time=False, shuffle_all=False)
    dataloader = DataLoader(datasets, batch_sampler=sampler, num_workers=4, collate_fn=collate_fn)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        text, mel_input, pos_text, pos_mel, text_lengths, mel_lengths = d


        text = text.to(DEVICE, non_blocking=True)
        mel_input = mel_input.to(DEVICE, non_blocking=True)
        pos_text = pos_text.to(DEVICE, non_blocking=True)
        pos_mel = pos_mel.to(DEVICE, non_blocking=True)
        text_lengths = text_lengths.to(DEVICE, non_blocking=True)
        #print(d[1].shape)
