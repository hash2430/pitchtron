from configs.hparams import create_hparams
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import librosa
from utils import read_wav_np
import os
hparams = create_hparams()
from scipy.io.wavfile import write
from utils import read_wav_np
import torch
def check_paths(lists, num_workers=16, tqdm=lambda x: x):
    files2d = [[] for i in range(len(lists))]
    futures = [[] for i in range(len(lists))]
    names = ['train', 'valid', 'test']
    for i in range(len(lists)):
        with open(lists[i],  'r', encoding='utf-8-sig') as f:
            files2d[i] = f.readlines()
    executor = ProcessPoolExecutor(max_workers=num_workers)

    for i in range(len(files2d)):
        for line in files2d[i]:
            path = line.split('|')[0]
            futures[i].append(executor.submit(partial(_process_utterance, path)))
        write_metadata([future.result() for future in tqdm(futures[i])], 'filelists', 'problematic_merge_korean_pron_{}.txt'.format(names[i]))

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
def _process_utterance(in_path):
    try:
        sr, wav = read_wav_np(in_path)
    except:
        return in_path+'\n'
    return ''






def write_metadata(metadata, out_dir, out_file):
    with open(os.path.join(out_dir, out_file), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write((m))