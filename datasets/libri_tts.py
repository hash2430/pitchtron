from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import numpy as np
from scipy.io import wavfile
from scipy import interpolate
from configs.hparams import create_hparams
hparams = create_hparams()
'''
change sampling rate of libritts from 24 kHz to 22.05 kHz
Never use this as is.trim is necessary.
'''
# TODO: trim with 60 top dB is required
def build_from_path(num_workers=16, tqdm=lambda x: x):
    train_file_list_file = 'filelists/libritts_train_clean_100_audiopath_text_sid_shorterthan10s_atleast5min_train_filelist.txt'
    val_file_list_file = 'filelists/libritts_train_clean_100_audiopath_text_sid_atleast5min_val_filelist.txt'
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures =[]
    with open(train_file_list_file, 'r', encoding='utf-8') as f:
        train_file_list = f.readlines()

    for line in train_file_list:
        in_path = line.split('|')[0]
        out_path = in_path.replace('train-clean-100', 'train-clean-100-22050')
        futures.append(executor.submit(partial(_process_utterance, in_path, out_path)))

    with open(val_file_list_file, 'r', encoding='utf-8') as f:
            val_file_list = f.readlines()

    for line in val_file_list:
        in_path = line.split('|')[0]
        out_path = in_path.replace('train-clean-100', 'train-clean-100-22050')
        futures.append(executor.submit(partial(_process_utterance, in_path, out_path)))

# I have not decided whether to separate dierectories for train/eval/test
# TODO: set hparams. max_wav_value=32768.0 because wav is preprocessed to be int16. Data type conversion required.
def _process_utterance(in_path, out_path):
    new_samplerate = hparams.sampling_rate
    old_samplerate, old_audio = wavfile.read(in_path)

    if old_samplerate != new_samplerate:
        dirname = os.path.dirname(out_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        duration = old_audio.shape[0] / old_samplerate

        time_old = np.linspace(0, duration, old_audio.shape[0])
        time_new = np.linspace(0, duration, int(old_audio.shape[0] * new_samplerate / old_samplerate))

        interpolator = interpolate.interp1d(time_old, old_audio.T)
        new_audio = interpolator(time_new).T

        wavfile.write(out_path, new_samplerate, np.round(new_audio).astype(old_audio.dtype))

