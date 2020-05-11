import matplotlib.pyplot as plt
import IPython.display as ipd

import sys
sys.path.append('waveglow/')

from itertools import cycle
import numpy as np
from scipy.io.wavfile import write
import pandas as pd
import librosa
import torch
from torch.utils.data import DataLoader

from configs.gst_200220 import create_hparams
from train import load_model
from waveglow.denoiser import Denoiser
from layers import TacotronSTFT
from data_utils import TextMelLoader, TextMelCollate
from text import cmudict, text_to_sequence
from pitchtron_utils import get_data_from_musicxml

hparams = create_hparams()
hparams.batch_size = 1
stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax)
speaker = "fy01"
checkpoint_path = '/mnt/sdc1/pitchtron/gst_200220/checkpoint_150500'
    # "models/pitchtron_libritts.pt"
pitchtron = load_model(hparams).cuda().eval()
pitchtron.load_state_dict(torch.load(checkpoint_path)['state_dict'])
waveglow_path = '/home/admin/projects/pitchtron_init_with_single/models/waveglow_256channels_v4.pt'
waveglow = torch.load(waveglow_path)['model'].cuda().eval()
denoiser = Denoiser(waveglow).cuda().eval()
arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')
audio_paths = 'filelists/tmp.txt'
test_set = TextMelLoader(audio_paths, hparams)
datacollate = TextMelCollate(1)
dataloader = DataLoader(test_set, num_workers=1, shuffle=False,batch_size=hparams.batch_size, pin_memory=False,
                        drop_last=False, collate_fn = datacollate)
speaker_ids = TextMelLoader("filelists/wav_less_than_12s_158_speakers_train.txt", hparams).speaker_ids
speaker_id = torch.LongTensor([speaker_ids[speaker]]).cuda()

for i, batch in enumerate(dataloader):
    reference_speaker = test_set.audiopaths_and_text[i][2]
    # x: (text_padded, input_lengths, mel_padded, max_len,
    #                  output_lengths, speaker_ids, f0_padded),
    # y: (mel_padded, gate_padded)
    x, y = pitchtron.parse_batch(batch)
    text_encoded = x[0]
    mel = x[2]
    pitch_contour = x[6]

    # inputs = text, style_input, speaker_ids, f0s
    with torch.no_grad():
        input = (text_encoded, mel, speaker_id, pitch_contour)
        mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = pitchtron.inference(input)

    with torch.no_grad():
        audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:, 0]
        audio = audio.squeeze(1).cpu().numpy()
        top_db=25
        for j in range(len(audio)):
            wav, _ = librosa.effects.trim(audio[j], top_db=top_db, frame_length=2048, hop_length=512)
            write("/mnt/sdc1/pitchtron_experiment/sample-{:03d}_target_speaker-{}_referende_speaker-{}-gst-150500.wav".format(i*hparams.batch_size+j, speaker, reference_speaker), hparams.sampling_rate, wav)
