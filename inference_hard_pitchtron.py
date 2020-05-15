'''
 This inference code normalizes f0 by
 1. Subtracting reference f0 mean from reference pitch contour
 2. Divide reference pitch contour with reference speaker f0 variance obtained by 'max-min'.
 3. Multiply reference pitch contour with target speaker f0 variance
 4. Add pitch contour with target speaker f0 mean
(4-1. If necessary, scale the target f0 mean with 1.1 scalar multiplication for adding)
 5. In case - ref_f0 + target_f0 < 0, target minimum f0 is used instead.
'''

''' 
TODO: Fitting reference f0 contour into target speaker vocal range (min+alpha, max-beta) by scaling would give more natural result.
High variance from reference signal gives unnatural sounding result
'''
import sys
sys.path.append('waveglow/')

from scipy.io.wavfile import write
import librosa
import torch
from torch.utils.data import DataLoader

from configs.grl_200224 import create_hparams
from train import load_model
from waveglow.denoiser import Denoiser
from layers import TacotronSTFT
from data_utils import TextMelLoader, TextMelCollate
from text import cmudict

hparams = create_hparams()
hparams.batch_size = 1
stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax)
# speaker = "fv02"
checkpoint_path ='/mnt/sdc1/pitchtron/grl_200224/checkpoint_291000'
f0s_meta_path = '/mnt/sdc1/pitchtron/single_init_200123/f0s_combined.txt'
    # "models/pitchtron_libritts.pt"
pitchtron = load_model(hparams).cuda().eval()
pitchtron.load_state_dict(torch.load(checkpoint_path)['state_dict'])
waveglow_path = '/home/admin/projects/pitchtron_init_with_single/models/waveglow_256channels_v4.pt'
waveglow = torch.load(waveglow_path)['model'].cuda().eval()
denoiser = Denoiser(waveglow).cuda().eval()
arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')
audio_paths = 'data/examples_pfp_single_sample.txt'
test_set = TextMelLoader(audio_paths, hparams)
datacollate = TextMelCollate(1)
dataloader = DataLoader(test_set, num_workers=1, shuffle=False,batch_size=hparams.batch_size, pin_memory=False,
                        drop_last=False, collate_fn = datacollate)
speaker_ids = TextMelLoader("filelists/wav_less_than_12s_158_speakers_train.txt", hparams).speaker_ids
# speaker_id = torch.LongTensor([speaker_ids[speaker]]).cuda()

# Load mean f0
with open(f0s_meta_path, 'r', encoding='utf-8-sig') as f:
    f0s_read = f.readlines()
f0s_mean = {}
f0s_min = {}
f0s_max = {}
f0s_var = {}
for i in range(len(f0s_read)):
    line = f0s_read[i].split('|')
    tmp_speaker = line[0]
    f0_mean = float(line[-1])
    f0s_mean[tmp_speaker] = f0_mean
    f0_min = float(line[1])
    f0s_min[tmp_speaker] = f0_min
    f0_max = float(line[2])
    f0s_max[tmp_speaker] = f0_max
    f0s_var[tmp_speaker] = f0_max - f0_min

for key, value in speaker_ids.items():
    speaker = key
    target_speaker = speaker
    target_speaker_f0_mean = f0s_mean[target_speaker]
    speaker_id = torch.LongTensor([value]).cuda()
    for i, batch in enumerate(dataloader):
        reference_speaker = test_set.audiopaths_and_text[i][2]
        reference_speaker_f0_mean = f0s_mean[reference_speaker]
        # x: (text_padded, input_lengths, mel_padded, max_len,
        #                  output_lengths, speaker_ids, f0_padded),
        # y: (mel_padded, gate_padded)
        x, y = pitchtron.parse_batch(batch)
        text_encoded = x[0]
        mel = x[2]
        pitch_contour = x[6]
        # normalize f0 for voiced frames
        mask = pitch_contour != 0.0
        pitch_contour[mask] -= reference_speaker_f0_mean
        pitch_contour[mask] /= f0s_var[reference_speaker]
        pitch_contour[mask] *= f0s_var[target_speaker]
        pitch_contour[mask] += target_speaker_f0_mean

        # take care of negative f0s when reference is female and target is male
        mask = pitch_contour < 0.0
        pitch_contour[mask] = f0s_min[target_speaker]
        tmp_nd = pitch_contour.cpu().numpy()
        tmp_mask = mask.cpu().numpy()



        with torch.no_grad():
            # get rhythm (alignment map) using tacotron 2
            mel_outputs, mel_outputs_postnet, gate_outputs, rhythm, reference_speaker_predicted1= pitchtron.forward(x)
            rhythm = rhythm.permute(1, 0, 2)

            # Using mel as input is not generalizable. I hope there is generalizable inference method as well.
            mel_outputs, mel_outputs_postnet, gate_outputs, _, reference_speaker_predicted2 = pitchtron.inference_noattention(
                (text_encoded, mel, speaker_id, pitch_contour, rhythm))

        with torch.no_grad():
            audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:, 0]
            audio = audio.squeeze(1).cpu().numpy()
            top_db=25
            for j in range(len(audio)):
                wav, _ = librosa.effects.trim(audio[j], top_db=top_db, frame_length=2048, hop_length=512)
                write("/mnt/sdc1/pitchtron_experiment/different_speaker_subjective_test/grl_002/{}/sample-{:03d}_target-{}_refer-{}-grl002-relative-rescaled-f0.wav".format(reference_speaker, i * hparams.batch_size + j, speaker, reference_speaker), hparams.sampling_rate, wav)