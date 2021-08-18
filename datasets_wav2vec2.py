# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
#import librosa
import soundfile as sf
from operator import itemgetter
from typing import Optional
import collections
import os
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
import sentencepiece as spm
from tqdm import tqdm
import math

from transformers import Wav2Vec2Processor

class TrainDatasets(Dataset):
    """
    Dataset class.
    """
    def __init__(self, csv_file, hp):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.
        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='\|', header=None)
        self.hp = hp
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.hp.spm_model)
        ## TODO: variable
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-lv60")

        ## TODO
        if self.hp.lengths_file is None or not os.path.exists(self.hp.lengths_file):
            print('lengths_file is not exists. Make...')
            lengths_list = []
            pbar = tqdm(range(len(self.landmarks_frame)))
            for idx in pbar:
                wav_name = self.landmarks_frame.loc[idx, 0]
                audio_input, sampling_rate = sf.read(wav_name)
                wav_input = self.processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_values
                ## TODO: check calucation for lengths (int(wav_input.shape[1]//320))
                # [1, lengths of wav] -> [lengths of wav]
                wav2vec2_length = math.floor((wav_input.shape[1] - 400) / 320.) + 1
                
                lengths_list.append(wav2vec2_length)
                
            self.lengths_np = np.array(lengths_list)
            np.save(self.hp.lengths_file, self.lengths_np)

    def __len__(self):                                                          
        return len(self.landmarks_frame)                                        
                                                                                
    def __getitem__(self, idx): 
        wav_name = self.landmarks_frame.loc[idx, 0]
        text_raw = self.landmarks_frame.loc[idx, 1].strip()
                                                                                
        textids = [self.sp.bos_id()] + self.sp.EncodeAsIds(text_raw) + [self.sp.eos_id()]
        text = np.array([int(t) for t in textids], dtype=np.int32)

        audio_input, sampling_rate = sf.read(wav_name)
        wav_input = self.processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_values
        ## TODO: check calucation for lengths (int(wav_input.shape[1]//320))
        # [1, lengths of wav] -> [lengths of wav]
        wav_input = wav_input.squeeze(0)
        wav2vec2_length = math.floor((wav_input.shape[0] - 400) / 320.) + 1

        text_length = len(text)
        pos_text = np.arange(1, text_length + 1)
        pos_wav2vec2 = np.arange(1, wav2vec2_length + 1)

        sample = {'text': text, 'text_length':text_length, 'wav_input':wav_input, 'wav2vec2_length':wav2vec2_length, 
                 'pos_wav2vec2':pos_wav2vec2, 'pos_text':pos_text}                               
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
    if isinstance(batch[0], collections.abc.Mapping):
        text = [d['text'] for d in batch]
        wav_input = [d['wav_input'] for d in batch]
        wav2vec2_lengths = [d['wav2vec2_length'] for d in batch]
        text_lengths = [d['text_length'] for d in batch]
        pos_wav2vec2 = [d['pos_wav2vec2'] for d in batch]
        pos_text = [d['pos_text'] for d in batch]
        
        text = _prepare_data(text).astype(np.int32)
        wav_input = _pad_wav(wav_input)
        pos_wav2vec2 = _prepare_data(pos_wav2vec2).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)

        return torch.LongTensor(text), torch.FloatTensor(wav_input), torch.LongTensor(pos_text), torch.LongTensor(pos_wav2vec2), \
               torch.LongTensor(text_lengths), torch.LongTensor(wav2vec2_lengths)


def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_wav(inputs):
    ## NOTE: change shapes
    _pad = 0
    def _pad_one(x, max_len):
        wav_len = x.shape[0]
        return np.pad(x, [0, max_len - wav_len], mode='constant', constant_values=_pad)
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
        if self.shuffle:
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

def main():
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
        text, wav_input, pos_text, pos_wav2vec2, text_lengths, wav2vec2_lengths = d

        text = text.to(DEVICE, non_blocking=True)
        wav_input = wav_input.to(DEVICE, non_blocking=True)
        pos_text = pos_text.to(DEVICE, non_blocking=True)
        pos_wav2vec2 = pos_wav2vec2.to(DEVICE, non_blocking=True)
        text_lengths = text_lengths.to(DEVICE, non_blocking=True)

if __name__ == '__main__':
    main()
