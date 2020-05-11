from plotting_utils import plot_spectrogram_to_numpy
from datasets.generate_mel_f0 import get_mel_and_f0
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
from configs.single_init_200123 import create_hparams
hparams = create_hparams()
import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)


big_E_path="/mnt/sdc1/mellotron_experiment/gst_token_weight_manipulation/mellotron/sample-t16-s34-000_target_speaker-fy17_referende_speaker-fy17-proposed800.wav"
small_E_path = "/mnt/sdc1/mellotron_experiment/gst_token_weight_manipulation/mellotron/sample-t16-s34-000_target_speaker-fy17_referende_speaker-fy17-proposed.wav"

mel1, _ = get_mel_and_f0(big_E_path, hparams.filter_length, hparams.hop_length, hparams.win_length, hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax, hparams.f0_min, hparams.f0_max, hparams.harm_thresh)
mel2, _ = get_mel_and_f0(small_E_path, hparams.filter_length, hparams.hop_length, hparams.win_length, hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax, hparams.f0_min, hparams.f0_max, hparams.harm_thresh)

def plot_spectrogram_to_numpy(spectrogram1, spectrogram2, save_name):
    ax1 = plt.subplot(2,1,1)
    im = ax1.imshow(spectrogram1, aspect="auto", origin="lower", cmap=plt.get_cmap('magma'),
                   interpolation='none')
    plt.colorbar(im, cax=ax1, ax=ax1)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    ax2 = plt.subplot(2, 1, 2)
    im2 = ax2.imshow(spectrogram2, aspect="auto", origin="lower", cmap=plt.get_cmap('magma'),
                   interpolation='none')
    plt.colorbar(im2, cax=ax2, ax=ax1)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    plt.show()
    plt.savefig(save_name)

def plot_spectrogram_to_numpy2(spectrogram1, spectrogram2, save_name):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(3,3),sharex=True,sharey=True)
    im = ax1.imshow(spectrogram1, vmin=-11.53, vmax=-0.5, aspect="auto", origin="lower", cmap=plt.get_cmap('magma'),
                   interpolation='none')
    plt.colorbar(im, ax=ax1)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    ax2 = plt.subplot(2, 1, 2)
    im2 = ax2.imshow(spectrogram2, vmin=-11.53, vmax=-0.5, aspect="auto", origin="lower", cmap=plt.get_cmap('magma'),
                   interpolation='none')
    plt.colorbar(im2, ax=ax2)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    plt.savefig(save_name)
if __name__ == '__main__':
    plot_spectrogram_to_numpy2(mel1.data.cpu().numpy(), mel2.data.cpu().numpy(), 'Single3.png')
