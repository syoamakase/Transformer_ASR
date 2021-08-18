## CONDIG
# Preprocess

# 48kHz
# n_mel_channels = 80
# segment_length = 16000
# pad_short = 2000
# filter_length = 1024
# hop_length = 512
# win_length = 2048
# sampling_rate = 48000
# mel_fmin = 70.0
# mel_fmax = 8000.0

# 24kHz
# n_mel_channels = 80
# segment_length = 16000
# pad_short = 2000
# filter_length = 1024
# hop_length = 256
# win_length = 1024
# sampling_rate = 24000
# mel_fmin = 70.0
# mel_fmax = 8000.0

# 22.05kHz
# n_mel_channels = 80
# segment_length = 16000
# pad_short = 2000
# filter_length = 1024
# hop_length = 256
# win_length = 1024
# sampling_rate = 22050
# mel_fmin = 70.0
# mel_fmax = 8000.0

# 16kHz
n_mel_channels = 40
segment_length = 16000
pad_short = 2000
filter_length = 400
hop_length = 160
win_length = 400
sampling_rate = 16000
mel_fmin = 0.0
mel_fmax = None
htk = True
norm = None
window = 'hamming'
preemp = True
