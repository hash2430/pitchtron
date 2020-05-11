from configs.hparams import create_hparams
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import librosa
from utils import read_wav_np
import os
hparams = create_hparams()
from scipy.io.wavfile import write
import torch
def build_from_path(in_dir, out_dir, filelist_names, spk_name_idx,num_workers=16, tqdm=lambda x: x):
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
            futures_val.append(executor.submit(partial(_process_utterance, pcm_file, out_path, spk_name_idx)))
        elif int(index) % 400 == 1:
            futures_test.append(executor.submit(partial(_process_utterance, pcm_file, out_path, spk_name_idx)))
        else:
            futures.append(executor.submit(partial(_process_utterance, pcm_file, out_path, spk_name_idx)))
        index += 1
    write_metadata([future.result() for future in tqdm(futures)], out_dir, filelist_names[0])
    write_metadata([future.result() for future in tqdm(futures_val)], out_dir, filelist_names[1])
    write_metadata([future.result() for future in tqdm(futures_test)], out_dir, filelist_names[2])
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
def _process_utterance(in_path, out_path, spk_name_idx):
    out_path = out_path.replace('pcm', 'wav')
    dir = os.path.dirname(out_path)
    # wav is saved as int 16
    command = 'sox -L -c 1 -e signed -b 16 -t raw -r 44100 {} -c 1 -e signed -b 16 -t wav -r 22050 {}'\
        .format(in_path, out_path)
    os.system(command)
    # int16 is converted into float32 here
    sampling_rate, audio = read_wav_np(out_path)
    wav, _ = librosa.effects.trim(audio, top_db=25, frame_length=2048, hop_length=512)
    wav = torch.from_numpy(wav).unsqueeze(0)
    wav = wav.squeeze(0).numpy()
    txt_file = in_path.replace('raw', 'script').replace('.pcm', '.pron')
    with open(txt_file, 'r', encoding='utf-8-sig') as f:
        line = f.readline()

    speaker = in_path.split('/')[spk_name_idx]
    write(out_path, 22050, wav)
    return (out_path, line.rstrip('\n'), speaker)



def write_metadata(metadata, out_dir, out_file):
    with open(os.path.join(out_dir, out_file), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')