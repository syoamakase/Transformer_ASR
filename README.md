# Transformer_ASR

This repository is for automatic speech recognition (ASR) using Transformer.

It includes the model architectures as follows:

- Conformer Network
- (Starndard) Transformer
- LSTM decoder
- RNN-T decoder (dev)

## Requirements (python >= 3.8.x, RECOMMEND: anaconda)

- torch
- tensorboard
- librosa
- sentencepiece

Since these packages are available by pip, please run `pip install -r requirements.txt`.

## Preprocess

Please clone https://github.com/syoamakase/examples
That repository includes (1) making acoustic features (2) making training script for ASR.


## Training

`python train.py --hp_file <path of hparams.py>`

## Test

`python test.py --load_name <path of network file>`
