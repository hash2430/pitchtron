from datasets.generate_mel_f0 import get_mel_and_f0
from librosa.core import dtw
from configs.as_is_200217 import create_hparams
import numpy as np
import torch
import datetime

out_file = 'Visualization/objective_measures_same_speaker/pfo_tones/out/12_samples_result_grl0.1_and_grl2.0.txt'
hparmas = create_hparams()
filter_length = hparmas.filter_length
hop_length = hparmas.hop_length
win_length = hparmas.win_length
n_mel_channels = hparmas.n_mel_channels
sampling_rate = hparmas.sampling_rate
mel_fmin = hparmas.mel_fmin
mel_fmax = hparmas.mel_fmax
f0_min = hparmas.f0_min
f0_max = hparmas.f0_max
harm_thresh = hparmas.harm_thresh
'''
TODO: Add energy-based prosody transfer evaluation metric
With GRL to reference embedding, it gives better energy similarity to the reference signal. 
Adding energy-based objective measure seems like a good idea.
'''
# How many sentences do I use for evaluations: No one says anything about it. How about 10 samples?

reference_file_list = 'Visualization/objective_measures_same_speaker/pfo_tones/reference.txt'
gst_file_list = 'Visualization/objective_measures_same_speaker/pfo_tones/gst.txt'
vlre_file_list = 'Visualization/objective_measures_same_speaker/pfo_tones/vlre_1D.txt'
proposed_file_list = 'Visualization/objective_measures_same_speaker/pfo_tones/grl_0_1.txt'
proposed_grl_file_list = 'Visualization/objective_measures_same_speaker/pfo_tones/grl_2_0.txt'

filelists = []
filelists.append(reference_file_list)
filelists.append(gst_file_list)
filelists.append(vlre_file_list)
filelists.append(proposed_file_list)
filelists.append(proposed_grl_file_list)
file_paths = [[] for i in range(len(filelists))]

for i in range(len(filelists)):
    with open(filelists[i], 'r', encoding='utf-8') as f:
        file_paths[i] = f.readlines()

for i in range(len(filelists)):
    for j in range(len(file_paths[0])):
        trimmed = file_paths[i][j].strip()
        file_paths[i][j] = trimmed


# Preprocessing: files[1] and files[2] needs dynamic time warping to compare with the reference signal

'''
1. Load wav
2. Calculate mel and F0
2. Dynamic time warp if
Return: GPE for gst, vlre, proposed, proposed_grl & numerator 
'''
# This is to see if loading 50 audios and saving the F0s to the RAM does not cause any trouble
def load():
    mels = [[] for i in range(len(filelists))]
    f0s = [[]for i in range(len(filelists))]
    for i in range(len(file_paths)):
        for j in range(len(file_paths[i])):
            mel, f0 = get_mel_and_f0(file_paths[i][j], filter_length, hop_length, win_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax, f0_min, f0_max, harm_thresh)
            mels[i].append(mel)
            f0s[i].append(f0)
    return mels, f0s

def DTW(mels, f0s):
    warped_true_mels, warped_gst_mels = align_dtw(mels[0], mels[1], 80)
    warped_true_mels2, warped_vlre_mels = align_dtw(mels[0], mels[2], 80)
    warped_true_mels3, warped_proposed_mels = align_dtw(mels[0], mels[3], 80)
    warped_true_mels4, warped_grl_mels = align_dtw(mels[0], mels[4], 80)

    warped_true_f0s, warped_gst_f0s = align_dtw(f0s[0], f0s[1], 1)
    warped_true_f0s2, warped_vlre_f0s = align_dtw(f0s[0], f0s[2], 1)
    warped_true_f0s3, warped_proposed_f0s = align_dtw(f0s[0], f0s[3], 1)
    warped_true_f0s4, warped_grl_f0s = align_dtw(f0s[0], f0s[4], 1)

    gst_mels = (warped_true_mels, warped_gst_mels)
    vlre_mels = (warped_true_mels2, warped_vlre_mels)
    proposed_mels = (warped_true_mels3, warped_proposed_mels)
    proposed_grl_mels = (warped_true_mels4, warped_grl_mels)

    gst_f0s = (warped_true_f0s, warped_gst_f0s) # First axis length is aligned with DTW. 0: reference 1: generated
    vlre_f0s = (warped_true_f0s2, warped_vlre_f0s)
    proposed_f0s = (warped_true_f0s3, warped_proposed_f0s)
    grl_f0s = (warped_true_f0s4, warped_grl_f0s)

    non_padded_f0_lens = [[] for i in range(4)]
    non_padded_f0_lens[0] = [i.shape[0] for i in warped_true_f0s]
    non_padded_f0_lens[1] = [i.shape[0] for i in warped_true_f0s2]
    non_padded_f0_lens[2] = [i.shape[0] for i in warped_true_f0s3]
    non_padded_f0_lens[3] = [i.shape[0] for i in warped_true_f0s4]

    non_padded_mel_lens = [[] for i in range(4)]
    non_padded_mel_lens[0] = [i.shape[0] for i in warped_true_mels]
    non_padded_mel_lens[1] = [i.shape[0] for i in warped_true_mels2]
    non_padded_mel_lens[2] = [i.shape[0] for i in warped_true_mels3]
    non_padded_mel_lens[3] = [i.shape[0] for i in warped_true_mels4]
    return gst_mels, vlre_mels, proposed_mels, proposed_grl_mels, gst_f0s, vlre_f0s, proposed_f0s, grl_f0s, non_padded_mel_lens, non_padded_f0_lens

def align_dtw(s1, s2, dim):
    warped_sequences1 = []
    warped_sequences2 = []
    for true, gst in zip(s1, s2):
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
        warped_sequences1.append(warped_true_mel)
        warped_sequences2.append(warped_gst_mel)
    return warped_sequences1, warped_sequences2 #(B, T, dim) not padded yet

def gpe(f0_target, f0_out):
    # pad for batch calculation
    lens = [f0_out_.shape[0] for f0_out_ in f0_out]
    max_len = max(lens)
    for i in range(len(f0_out)):
        if lens[i] < max_len:
            pad_width = [[0, max_len - lens[i]], [0, 0]]
            tmp = np.pad(f0_out[i], pad_width)
            f0_out[i] = tmp
            tmp = np.pad(f0_target[i], pad_width)
            f0_target[i] = tmp

    f0_out = torch.from_numpy(np.stack(f0_out)).squeeze()
    f0_target = torch.from_numpy(np.stack(f0_target)).squeeze()
    out_voiced_mask = f0_out != 0
    tmp1 = out_voiced_mask.cpu().numpy()
    target_voiced_mask = f0_target != 0
    tmp2 = target_voiced_mask.cpu().numpy()
    diff_abs = (f0_out - f0_target).abs()
    # tmp3 = diff_abs.cpu().numpy()
    erronous_prediction_mask = diff_abs > (0.2 * f0_target)
    tmp4 = erronous_prediction_mask.cpu().numpy()

    numerator = out_voiced_mask * target_voiced_mask * erronous_prediction_mask
    tmp5 = numerator.cpu().numpy()
    denominator = out_voiced_mask * target_voiced_mask
    tmp6 = denominator.cpu().numpy()

    numerator = torch.FloatTensor([numerator.sum(dim=(0,1))])
    denominator = torch.FloatTensor([denominator.sum(dim=(0,1))])
    loss = numerator / (denominator+1e-3)
    return loss

def GPE(gst_f0s, vlre_f0s, proposed_f0s, grl_f0s):
    # get GPE for gst
    gpe_gst = gpe(gst_f0s[0], gst_f0s[1])

    # get GPE for vlre
    gpe_vlre = gpe(vlre_f0s[0], vlre_f0s[1])

    gpe_proposed = gpe(proposed_f0s[0], proposed_f0s[1])

    # get GPE for proposed_grl
    gpe_proposed_grl = gpe(grl_f0s[0], grl_f0s[1])

    return gpe_gst, gpe_vlre, gpe_proposed, gpe_proposed_grl

def vde(f0_target, f0_out, non_padded_lens):
    # Non-padded true sequence length after dtw is required for denominator
    lens = [f0_out_.shape[0] for f0_out_ in f0_out]

    # Pad for batch calculation
    max_len = max(lens)
    for i in range(len(f0_out)):
        if lens[i] < max_len:
            pad_width = [[0, max_len - lens[i]], [0, 0]]
            f0_out[i] = np.pad(f0_out[i], pad_width)
            f0_target[i] = np.pad(f0_target[i], pad_width)

    f0_out = torch.from_numpy(np.stack(f0_out)).squeeze()
    f0_target = torch.from_numpy(np.stack(f0_target)).squeeze()

    out_voicing_decision = f0_out != 0
    target_voicing_decision = f0_target != 0

    mismatched_voicing_decision_mask = out_voicing_decision != target_voicing_decision
    numerator = mismatched_voicing_decision_mask.numpy()
    numerator = torch.FloatTensor([numerator.sum()])

    denominator = torch.FloatTensor(non_padded_lens).numpy()
    denominator = torch.FloatTensor([denominator.sum()])

    loss = numerator / denominator
    return loss


def VDE(gst_f0s, vlre_f0s, proposed_f0s, grl_f0s, non_padded_lens):
    #  get VDE for gst
    vde_gst = vde(gst_f0s[0], gst_f0s[1], non_padded_lens[0])

    # get VDE for vlre
    vde_vlre = vde(vlre_f0s[0], vlre_f0s[1], non_padded_lens[1])

    # get VDE for proposed
    vde_proposed = vde(proposed_f0s[0], proposed_f0s[1], non_padded_lens[2])

    # get VDE for grl
    vde_grl = vde(grl_f0s[0], grl_f0s[1], non_padded_lens[3])
    return vde_gst, vde_vlre, vde_proposed, vde_grl

def ffe(f0_target, f0_out, non_padded_lens):
    lens = [f0_out_.shape[0] for f0_out_ in f0_out]
    max_len = max(lens)
    for i in range(len(f0_out)):
        if lens[i] < max_len:
            pad_width = [[0, max_len - lens[i]], [0, 0]]
            tmp = np.pad(f0_out[i], pad_width)
            f0_out[i] = tmp
            tmp = np.pad(f0_target[i], pad_width)
            f0_target[i] = tmp

    f0_out = torch.from_numpy(np.stack(f0_out)).squeeze()
    f0_target = torch.from_numpy(np.stack(f0_target)).squeeze()
    out_voiced_mask = f0_out != 0
    tmp1 = out_voiced_mask.cpu().numpy()
    target_voiced_mask = f0_target != 0
    tmp2 = target_voiced_mask.cpu().numpy()
    diff_abs = (f0_out - f0_target).abs()
    # tmp3 = diff_abs.cpu().numpy()
    erronous_prediction_mask = diff_abs > 0.2 * f0_target
    tmp4 = erronous_prediction_mask.cpu().numpy()

    # denominator = torch.FloatTensor([f0_target.shape[0]])
    denominator = torch.FloatTensor(non_padded_lens).numpy()
    denominator = torch.FloatTensor([denominator.sum()])
    numerator1 = out_voiced_mask * target_voiced_mask * erronous_prediction_mask
    numerator1 = numerator1.sum()
    numerator2 = out_voiced_mask != target_voiced_mask
    numerator2 = numerator2.sum()
    numerator = torch.FloatTensor([numerator1 + numerator2])
    loss = numerator / (
        denominator)  # removed adding 1e-3 to denominator because it seems unlikely for denominator to be zero
    return loss

def FFE(gst_f0s, vlre_f0s, proposed_f0s, grl_f0s, non_padded_lens):
    ffe_gst = ffe(gst_f0s[0], gst_f0s[1], non_padded_lens[0])
    ffe_vlre = ffe(vlre_f0s[0], vlre_f0s[1], non_padded_lens[1])
    ffe_proposed = ffe(proposed_f0s[0], proposed_f0s[1], non_padded_lens[2])
    ffe_grl = ffe(grl_f0s[0], grl_f0s[1], non_padded_lens[3])

    return ffe_gst, ffe_vlre, ffe_proposed, ffe_grl

def mcd(target_mels, out_mels, true_lens):
    # MCD13: mse along 13 dims. Exclude 0th mel to make it indifferent of overall energy scale.
    # Use unpadded true lens for denominator
    lens = [target_mels_.shape[0] for target_mels_ in target_mels]
    max_len = max(lens)
    for i in range(len(target_mels)):
        if lens[i] < max_len:
            pad_width = [[0, max_len - lens[i]], [0, 0]]
            target_mels[i] = np.pad(target_mels[i], pad_width)
            out_mels[i] = np.pad(out_mels[i], pad_width)

    out_mels = torch.from_numpy(np.stack(out_mels)).squeeze()[:,:,1:14]
    target_mels = torch.from_numpy(np.stack(target_mels)).squeeze()[:,:,1:14]
    diff = out_mels - target_mels
    diff_sq = diff**2
    tmp = diff_sq.sum(dim=-1).squeeze()
    tmp = torch.sqrt(tmp)
    tmp = tmp.sum()
    numerator = tmp
    denominator = torch.FloatTensor([sum(true_lens)])
    mcd = numerator / denominator
    # Google's work does not multiply K
    # "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron"
    # K = 10 / np.log(10) * np.sqrt(2)
    return mcd

def MCD(gst_mels, vlre_mels, proposed_mels, proposed_grl_mels, non_padded_mel_lens):
    gst_mcd = mcd(gst_mels[0], gst_mels[1], non_padded_mel_lens[0])
    vlre_mcd = mcd(vlre_mels[0], vlre_mels[1], non_padded_mel_lens[1])
    proposed_mcd = mcd(proposed_mels[0], proposed_mels[1], non_padded_mel_lens[2])
    grl_mcd = mcd(proposed_grl_mels[0], proposed_grl_mels[1], non_padded_mel_lens[3])
    return gst_mcd, vlre_mcd, proposed_mcd, grl_mcd

if __name__ == '__main__':
    start = datetime.datetime.now()

    mels, f0s = load() # list of list of tensors
    gst_mels, vlre_mels, proposed_mels, proposed_grl_mels, gst_f0s, vlre_f0s, proposed_f0s, grl_f0s, non_padded_mel_lens, non_padded_f0_lens = DTW(mels, f0s)
    GPEs = GPE(gst_f0s, vlre_f0s, proposed_f0s, grl_f0s)
    VDEs = VDE(gst_f0s, vlre_f0s, proposed_f0s, grl_f0s, non_padded_f0_lens)
    FFEs = FFE(gst_f0s, vlre_f0s, proposed_f0s, grl_f0s, non_padded_f0_lens)
    MCDs = MCD(gst_mels, vlre_mels, proposed_mels, proposed_grl_mels, non_padded_mel_lens)

    print("          {:^10} {:^10} {:^10} {:^10}".format('GPE', 'VDE', 'FFE', 'MCD'))
    print("gst       {:7.2f}% {:7.2f}% {:7.2f}% {:5.2f} dB".format(GPEs[0].item()*100, VDEs[0].item()*100, FFEs[0].item()*100, MCDs[0].item()))
    print("vlre_1D   {:7.2f}% {:7.2f}% {:7.2f}% {:5.2f} dB".format(GPEs[1].item() * 100, VDEs[1].item() * 100,
                                                                        FFEs[1].item() * 100, MCDs[1].item()))
    print("pitchtron {:7.2f}% {:7.2f}% {:7.2f}% {:5.2f} dB".format(GPEs[2].item() * 100, VDEs[2].item() * 100,
                                                                            FFEs[2].item() * 100, MCDs[2].item()))
    print("grl_0.08  {:7.2f}% {:7.2f}% {:7.2f}% {:5.2f} dB".format(GPEs[3].item() * 100, VDEs[3].item() * 100,
                                                                            FFEs[3].item() * 100, MCDs[3].item()))
    # print("    {:^10} {:^10} {:^10} {:^10}".format("GST", "VLRE", "PROPOSED", "P_GRL 0.02"))
    # print("GPEs: {:7.2f}%, {:7.2f}%, {:7.2f}%, {:7.2f}%\n".format(GPEs[0].item()*100, GPEs[1].item()*100, GPEs[2].item()*100, GPEs[3].item()*100))
    # print("VDEs: {:7.2f}%, {:7.2f}%, {:7.2f}%, {:7.2f}%\n".format(VDEs[0].item()*100, VDEs[1].item()*100, VDEs[2].item()*100, VDEs[3].item()*100))
    # print("FFEs: {:7.2f}%, {:7.2f}%, {:7.2f}%, {:7.2f}%\n".format(FFEs[0].item()*100, FFEs[1].item()*100, FFEs[2].item()*100, FFEs[3].item()*100))
    # print("MCDs: {:5.2f} dB, {:5.2f} dB, {:5.2f} dB, {:5.2f} dB\n".format(MCDs[0].item(), MCDs[1].item(), MCDs[2].item(), MCDs[3].item()))
    end = datetime.datetime.now()
    spent_time = end - start
    print("Time spent: {}".format(spent_time))
    with open(out_file, 'w') as f:
        # f.write("    {:^10} {:^10} {:^10} {:^10}\n".format("GST", "VLRE", "PROPOSED", "P_GRL 0.02"))
        # f.write("GPEs: {:7.2f}%, {:7.2f}%, {:7.2f}%, {:7.2f}%\n".format(GPEs[0].item()*100, GPEs[1].item()*100, GPEs[2].item()*100, GPEs[3].item()*100))
        # f.write("VDEs: {:7.2f}%, {:7.2f}%, {:7.2f}%, {:7.2f}%\n".format(VDEs[0].item()*100, VDEs[1].item()*100, VDEs[2].item()*100, VDEs[3].item()*100))
        # f.write("FFEs: {:7.2f}%, {:7.2f}%, {:7.2f}%, {:7.2f}%\n".format(FFEs[0].item()*100, FFEs[1].item()*100, FFEs[2].item()*100, FFEs[3].item()*100))
        # f.write("MCDs: {:5.2f} dB, {:5.2f} dB, {:5.2f} dB, {:5.2f} dB\n".format(MCDs[0].item(), MCDs[1].item(), MCDs[2].item(), MCDs[3].item()))
        # f.write("Time spent: {}".format(spent_time))
        f.write(gst_file_list)
        f.write(vlre_file_list)
        f.write(proposed_file_list)
        f.write(proposed_grl_file_list)
        f.write("          {:^10} {:^10} {:^10} {:^10}\n".format('GPE', 'VDE', 'FFE', 'MCD'))
        f.write("gst       {:7.2f}% {:7.2f}% {:7.2f}% {:5.2f} dB\n".format(GPEs[0].item() * 100,
                                                                                VDEs[0].item() * 100,
                                                                                FFEs[0].item() * 100, MCDs[0].item()))
        f.write("vlre_1D   {:7.2f}% {:7.2f}% {:7.2f}% {:5.2f} dB\n".format(GPEs[1].item() * 100,
                                                                                VDEs[1].item() * 100,
                                                                                FFEs[1].item() * 100, MCDs[1].item()))
        f.write("grl_0.1 {:7.2f}% {:7.2f}% {:7.2f}% {:5.2f} dB\n".format(GPEs[2].item() * 100,
                                                                                VDEs[2].item() * 100,
                                                                                FFEs[2].item() * 100, MCDs[2].item()))
        f.write("grl_2.0   {:7.2f}% {:7.2f}% {:7.2f}% {:5.2f} dB\n".format(GPEs[3].item() * 100,
                                                                                VDEs[3].item() * 100,
                                                                                FFEs[3].item() * 100, MCDs[3].item()))
        f.write("Time spent: {}".format(spent_time))