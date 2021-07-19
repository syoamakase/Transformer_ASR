import argparse
import os
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-n', '--num_speakers')
    args = parser.parse_args()
    filename = args.filename
    num_speakers = args.num_speakers

    with open(filename) as f:
        for line in f:
            line = line.strip()
            random_id = np.random.randint(num_speakers)
            print(f'{line}|{random_id}')
