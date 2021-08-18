import argparse
import os

parser = argparse.ArgumentParser() 
parser.add_argument('-S', '--script_filename', required=True)
parser.add_argument('--base_path', default='data/train/mel/')
parser.add_argument('--ext', default='htk')
args = parser.parse_args()
script_filename = args.script_filename
ext = args.ext
base_path = args.base_path

with open(script_filename) as f:
    for line in f:
        file_id, text = line.strip().split('|',1)
        speaker_id = file_id.split('_')[0]
        file_name = os.path.join(base_path, speaker_id, f'{file_id}.{ext}')
        print(f'{os.path.abspath(file_name)}|{text}')
