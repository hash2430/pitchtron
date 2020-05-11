from configs.hparams import create_hparams
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
hparams = create_hparams()
'''
Never use this as is.
This script convert pcm to wav22050, however trimming is not done.
This script takes lbl as input.
text scripts does not support lbl so this script is useless for now.
'''
#TODO: add trimming, add support for lbl phoneme representation.(it seems useless)
def build_from_path(in_dir, out_dir, num_workers=16, tqdm=lambda x: x):
    pcm_files = []
    # for all speakers, count index and either add to train_list/eval_list/test_list
    speakers = os.listdir(in_dir)
    for speaker in speakers:
        path = os.path.join(in_dir, speaker, 'raw')
        pcms = os.listdir(path)
        for pcm in pcms:
            pcm_files.append(os.path.join(path, pcm))

    for speaker in speakers:
        path = os.path.join(in_dir, speaker, 'wav_22050')
        if not os.path.exists(path):
            os.makedirs(path)
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    futures_val = []
    futures_test = []
    index = 1
    for pcm_file in pcm_files:
        out_path = pcm_file.replace('raw', 'wav_22050')
        if int(index) % 400 == 0:
            futures_val.append(executor.submit(partial(_process_utterance, pcm_file, out_path)))
        elif int(index) % 400 == 1:
            futures_test.append(executor.submit(partial(_process_utterance, pcm_file, out_path)))
        else:
            futures.append(executor.submit(partial(_process_utterance, pcm_file, out_path)))
        index += 1
    write_metadata([future.result() for future in tqdm(futures)], out_dir, 'train_file_list.txt')
    write_metadata([future.result() for future in tqdm(futures_val)], out_dir, 'valid_file_list.txt')
    write_metadata([future.result() for future in tqdm(futures_test)], out_dir, 'test_file_list.txt')
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
# TODO: set hparams. max_wav_value=32768.0 because wav is preprocessed to be int16
# This does not have resulting directory yet. It was overrided by selvas_multispeaker_pron
def _process_utterance(in_path, out_path):
    out_path = out_path.replace('pcm', 'wav')
    dir = os.path.dirname(out_path)
    command = 'sox -L -c 1 -e signed -b 16 -t raw -r 44100 {} -c 1 -e signed -b 16 -t wav -r 22050 {}'\
        .format(in_path, out_path)
    os.system(command)
    txt_file = in_path.replace('raw', 'lab').replace('.pcm', '.lbl')
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    phonemes = [line.strip().split('\t')[1] for line in lines]
    text = "♡"+"♡".join(phonemes)
    speaker = in_path.split('/')[5]
    return (out_path, text, speaker)



def write_metadata(metadata, out_dir, out_file):
    with open(os.path.join(out_dir, out_file), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')