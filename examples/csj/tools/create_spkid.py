import argparse
import os
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('spk2utt')
    args = parser.parse_args()
    filename = args.filename
    spk2utt = args.spk2utt

    spk2utt_dict = {}
    num_speaker = 0
    with open(spk2utt) as f:
        for line in f:
            line = line.strip().split(' ')
            for spkid in line[1:]:
                spk2utt_dict[spkid] = num_speaker
            num_speaker += 1

    with open(filename) as f:
        for line in f:
            line = line.strip().split('|')
            file_id, _ = os.path.splitext(os.path.basename(line[0]))
            print(f'{line[0]}|{line[1]}|{spk2utt_dict[file_id]}|{line[2]}')
