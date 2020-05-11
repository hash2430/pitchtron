from datasets.generate_mel_f0 import get_mel_and_f0
import matplotlib.pyplot as plt
from configs.grl_200224 import create_hparams
from librosa.core import dtw
import numpy as np
hparams = create_hparams()
reference_path = '/mnt/sdd1/selvas_wav/pfo/wav_trimmed_22050/pfo00036.wav'
original_path = "/mnt/sdc1/mellotron_experiment/linear_manipulation/vlre_1D/sample-000_target_speaker-pfo_referende_speaker-pfo-vlre_1D_original.wav"
half_path = "/mnt/sdc1/mellotron_experiment/linear_manipulation/vlre_1D/sample-000_target_speaker-pfo_referende_speaker-pfo-vlre_1D_0.5.wav"
def align_dtw(true, gst, dim):
    _, idx = dtw(X=true, Y=gst, backtrack=True)
    idx_t = idx.transpose()
    true_idx = np.flip(idx_t[0])
    gst_idx = np.flip(idx_t[1])
    true = true.transpose(0, 1)
    gst = gst.transpose(0, 1)
    warped_true_mel = np.zeros((len(idx_t[0]), dim))
    warped_gst_mel = np.zeros((len(idx_t[0]), dim))
    for i in range(len(idx_t[0])):
        warped_true_mel[i] = true[true_idx[i]]
        warped_gst_mel[i] = gst[gst_idx[i]]
    return warped_true_mel, warped_gst_mel #(B, T, dim) not padded yet

# open wav and extract f0
_, reference_f0 = get_mel_and_f0(reference_path, hparams.filter_length, hparams.hop_length, hparams.win_length, hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax, hparams.f0_min, hparams.f0_max, hparams.harm_thresh)
_, original_f0 = get_mel_and_f0(original_path, hparams.filter_length, hparams.hop_length, hparams.win_length, hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax, hparams.f0_min, hparams.f0_max, hparams.harm_thresh)
_, rescaled_f0 = get_mel_and_f0(half_path, hparams.filter_length, hparams.hop_length, hparams.win_length, hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax, hparams.f0_min, hparams.f0_max, hparams.harm_thresh)

reference_f0_aligned1, original_f0 = align_dtw(reference_f0, original_f0, 1)
reference_f0_aligned2, rescaled_f0 = align_dtw(reference_f0*0.5, rescaled_f0, 1)


reference_f0 = reference_f0_aligned1.squeeze()
original_f0 = original_f0.squeeze()
rescaled_f0 = rescaled_f0.squeeze()

x_reference = range(len(reference_f0))
x_gst = range(len(original_f0))
x_proposed = range(len(rescaled_f0))

plt.plot(x_reference, reference_f0, label='Reference')
plt.plot(x_gst, original_f0, label='generated')
plt.plot(x_proposed, rescaled_f0, label='rescaled')

plt.xlabel('Frames')
plt.ylabel('F0s')

plt.title('Pitch contour comparison')
plt.legend()
plt.show()