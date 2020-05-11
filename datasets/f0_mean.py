from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import librosa
from utils import read_wav_np, load_wav_to_torch
import os
from scipy.io.wavfile import write
import torch
import glob
from scipy import interpolate
from yin import compute_yin
from random import shuffle


def get_f0(audio, sampling_rate=22050, frame_length=1024,
           hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
    f0, harmonic_rates, argmins, times = compute_yin(
        audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
        harm_thresh)
    pad = int((frame_length / hop_length) / 2)
    f0 = [0.0] * pad + f0 + [0.0] * pad

    f0 = np.array(f0, dtype=np.float32)
    return f0

def build_from_path(root, hparams, num_workers=16, tqdm=lambda x: x):
    speakers = glob.glob(os.path.join(root,'*'))
    speakers.sort()
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for speaker in speakers:
        new_root = speaker
        futures.append(executor.submit(partial(_process_speaker, new_root, hparams)))
    out_file = os.path.join(root, 'f0s.txt')
    write_metadata([future.result() for future in tqdm(futures)], out_file)

def _process_speaker(root, hparams):
    # filelist = glob.glob(os.path.join(root, 'wav_22050','*.wav'))
    filelist = glob.glob(os.path.join(root, '*.wav'))
    shuffle(filelist)
    f0_sum_tot = 0
    min_tot = 1000
    max_tot = 0
    num_frames_tot = 0
    for i in range(10):
        filepath = filelist[i]
        audio, sampling_rate = load_wav_to_torch(filepath)
        f0 = get_f0(audio.cpu().numpy(), hparams.sampling_rate,
                         hparams.filter_length, hparams.hop_length, hparams.f0_min,
                         hparams.f0_max, hparams.harm_thresh)
        min_f0 = np.min(f0[np.nonzero(f0)])
        max_f0 = f0.max()
        if min_tot > min_f0:
            min_tot = min_f0
        if max_tot < max_f0:
            max_tot = max_f0
        sum_over_frames = np.sum(f0[np.nonzero(f0)])
        n_frames = len(f0[np.nonzero(f0)])
        f0_sum_tot += sum_over_frames
        num_frames_tot += n_frames
    f0_mean = f0_sum_tot / num_frames_tot
    speaker = os.path.basename(root)
    return speaker, round(min_tot), round(max_tot), round(f0_mean)

def write_metadata(metadata, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        for m in metadata:
            if m is None:
                continue
            f.write('|'.join([str(x) for x in m]) + '\n')