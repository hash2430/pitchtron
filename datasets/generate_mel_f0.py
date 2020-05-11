import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import torch

import layers
from utils import load_wav_to_torch
from yin import compute_yin


# define static get_mel_and_f0 to prevent lock contention on TextMelLoader object
def get_f0(audio, sampling_rate=22050, frame_length=1024,
           hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
    f0, harmonic_rates, argmins, times = compute_yin(
        audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
        harm_thresh)
    pad = int((frame_length / hop_length) / 2)
    f0 = [0.0] * pad + f0 + [0.0] * pad

    f0 = np.array(f0, dtype=np.float32)
    return f0
def get_mel_and_f0(filepath, filter_length, hop_length, win_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax, f0_min, f0_max, harm_thresh):
    stft = layers.TacotronSTFT(filter_length, hop_length, win_length,
            n_mel_channels, sampling_rate, mel_fmin,
            mel_fmax)
    audio, sampling_rate = load_wav_to_torch(filepath)
    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio
    # I changed them to float32 during preprocessing so this normalization is unnecessary.
    audio_norm = audio_norm.unsqueeze(0)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)

    f0 = get_f0(audio.cpu().numpy(), sampling_rate,
                filter_length, hop_length, f0_min,
                f0_max, harm_thresh)
    f0 = torch.from_numpy(f0)[None]
    f0 = f0[:, :melspec.size(1)]

    return melspec, f0

def build_from_path(lists, hparams, num_workers=16, tqdm=lambda x: x):
    files2d = [[] for i in range(len(lists))]
    futures = [[] for i in range(len(lists))]

    for i in range(len(lists)):
        with open(lists[i], 'r', encoding='utf-8-sig') as f:
            files2d[i] = f.readlines()

    executor = ProcessPoolExecutor(max_workers=num_workers)
    for i in range(len(files2d)):
        for line in files2d[i]:
            path = line.split('|')[0]
            futures[i].append(executor.submit(partial(_process_utterance, path, hparams.filter_length, hparams.hop_length, hparams.win_length, hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax, hparams.f0_min, hparams.f0_max, hparams.harm_thresh)))
        print([future.result() for future in tqdm(futures[i])])

'''
1. Read each file
2. Down sample to 22050Hz
3. Create meta file with the format 'path|phonemes|speaker'
["g",  "n",  "d",  "l",  "m",  "b",  "s",  "-", "j",   "q",
                "k", "t", "p", "h", "x", "w", "f", "c", "z", "A",
                "o", "O", "U", "u", "E", "a", "e", "1", "2", "3",
                "4", "5", "6", "7", "8", "9", "[", "]", "<", ">",
                "G", "N", "D", "L", "M", "B", "0", "K", ";;",";", "sp", "*",
                "$", "?", "!","#"]
'''
# I have not decided whether to separate dierectories for train/eval/test
def _process_utterance(in_path, filter_length, hop_length, win_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax, f0_min, f0_max, harm_thresh):
    # Move this to build_from_path
    mel_out_path = in_path.replace('wav_22050', 'mel').replace('.wav','.pt')
    f0_out_path = in_path.replace('wav_22050', 'f0').replace('.wav','.pt')
    dir = os.path.dirname(mel_out_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.dirname(f0_out_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # int16 is converted into float32 here
    mel, f0 = get_mel_and_f0(in_path, filter_length, hop_length, win_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax, f0_min, f0_max, harm_thresh)
    torch.save(mel, mel_out_path)
    torch.save(f0, f0_out_path)
    return
