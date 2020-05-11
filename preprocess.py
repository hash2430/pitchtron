import argparse
import os
from tqdm import tqdm
from datasets import libri_tts, selvas_multi_lbl,selvas_multispeaker_pron, public_korean_pron, check_file_integrity, generate_mel_f0, f0_mean
from configs.korean_200113 import create_hparams

hparams = create_hparams()

# WARN: Do not use this without adding trim
# def preprocess_libri_tts(args):
#     libri_tts.build_from_path(args.num_workers, tqdm=tqdm)

# WARN: Do not use this without adding trim and supporting lbl phoneme sets
# def preprocess_selvas_multi(args):
#     in_dir = '/past_projects/DB/selvasai/selvasai_organized'
#     out_dir = 'filelists'
#     selvas_multi_lbl.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)

def preprocess_selvas_multispeaker_pron(args):
    # in_dir = '/past_projects/DB/selvasai/selvasai_organized'
    in_dir = '/mnt/sdd1/leftout_males'
    # in_dir = '/mnt/sdd1/selvas_emotion'
    out_dir = 'filelists'
    # in order of train-valid-text
    filelists_name = [
        'train_file_list_pron_sub.txt',
        'valid_file_list_pron_sub.txt',
        'test_file_list_pron_sub.txt'
    ]
    selvas_multispeaker_pron.build_from_path(in_dir, out_dir, filelists_name, 4, args.num_workers, tqdm=tqdm)

# TODO: lang code is written in this procedure. Langcode==1 for korean-only case is hard-coded for now.
# TODO: This must be fixed to support english and other languages as well.
def _integrate(train_file_lists, target_train_file_list):
    sources = [[] for i in range(len(train_file_lists))]
    i = 0
    for file_list in train_file_lists:
        with open(file_list, 'r', encoding='utf-8-sig') as f:
            sources[i] = f.readlines()
        i += 1

    # integrate meta file
    lang_code = 1
    with open(target_train_file_list, 'w', encoding='utf-8-sig') as f:
        for i in range(len(sources)):
            for j in range(len(sources[i])):
                sources[i][j] = sources[i][j].rstrip() + '|{}\n'.format(str(lang_code))  # add language code

        for i in range(1, len(sources)):
            sources[0] += sources[i]

        # shuffle or not

        f.writelines(sources[0])

def preprocess_public_korean_pron(args):
    # in_dir = '/mnt/sdd1/korean_public'
    in_dir = '/mnt/sdd1/leftout_korean_old_male'
    out_dir = 'filelists'
    filelists_name = [
        'train_korean_pron.txt',
        'valid_korean_pron.txt',
        'test_korean_pron.txt'
    ]
    public_korean_pron.build_from_path(in_dir, out_dir, filelists_name, args.num_workers, tqdm=tqdm)

# This better not be done multithread because meta file is going to be locked and it will be inefficient.
def integrate_dataset(args):
    # train_file_lists = [
    #     'filelists/libritts_train_clean_100_audiopath_text_sid_shorterthan10s_atleast5min_train_filelist.txt',
    #     'filelists/train_file_list.txt']
    # eval_file_lists = [
    #     '/home/administrator/projects/mellotron/filelists/libritts_train_clean_100_audiopath_text_sid_atleast5min_val_filelist.txt',
    #     '/home/administrator/projects/mellotron/filelists/valid_file_list.txt']
    # test_file_lists = [
    #     'filelists/libritts_train_clean_100_audiopath_text_sid_shorterthan10s_atleast5min_test_filelist.txt',
    #     'filelists/test_file_list.txt']
    #
    # target_train_file_list = 'filelists/libritts_selvas_multi_train.txt'
    # target_eval_file_list = 'filelists/libritts_selvas_multi_eval.txt'
    # target_test_file_list = 'filelists/libritts_selvas_multi_test.txt'

    train_file_lists = ['/home/administrator/projects/mellotron/filelists/train_file_list_pron.txt',
                        '/home/administrator/projects/mellotron/filelists/public_korean_train_file_list_pron.txt'
    ]

    eval_file_lists = ['/home/administrator/projects/mellotron/filelists/valid_file_list_pron.txt',
                       '/home/administrator/projects/mellotron/filelists/public_korean_valid_file_list_pron.txt'
    ]

    test_file_lists = ['/home/administrator/projects/mellotron/filelists/test_file_list_pron.txt',
                       '/home/administrator/projects/mellotron/filelists/public_korean_test_file_list_pron.txt'
    ]

    target_train_file_list = 'filelists/merge_korean_pron_train.txt'
    target_eval_file_list = 'filelists/merge_korean_pron_valid.txt'
    target_test_file_list = 'filelists/merge_korean_pron_test.txt'

    # merge train lists
    _integrate(train_file_lists, target_train_file_list)

    # merge eval lists
    _integrate(eval_file_lists, target_eval_file_list)

    # merge test lists
    _integrate(test_file_lists, target_test_file_list)

    print('Dataset integration has been complete')

# Try opening files on the filelist and write down the files with io error.
def check_for_file_integrity(args):
    lists = ['filelists/merge_korean_pron_train.txt', 'filelists/merge_korean_pron_valid.txt', 'filelists/merge_korean_pron_test.txt']
    check_file_integrity.check_paths(lists, tqdm=tqdm)

def gen_mel_f0(args):
    lists = ['filelists/merge_korean_pron_train.txt', 'filelists/merge_korean_pron_valid.txt', 'filelists/merge_korean_pron_test.txt']
    generate_mel_f0.build_from_path(lists, hparams, tqdm=tqdm)

def preprocess_cal_f0_scale_per_training_speaker(args):
    # root = '/mnt/sdd1/selvas_emotion'
    # root = '/mnt/sdd1/leftout_males'
    # root = '/mnt/sdd1/leftout_korean_old_male/wav_22050'
    root = '/mnt/sdd1/korean_public/wav_22050'
    f0_mean.build_from_path(root, hparams, tqdm=tqdm)

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir', default=os.path.expanduser('/past_projects/DB'))
    # parser.add_argument('--output', default='sitec')
    parser.add_argument('--dataset', required=True,
                        choices=['blizzard', 'ljspeech', 'sitec', 'sitec_short', 'selvas_multi', 'libri_tts', 'selvas_multispeaker_pron',
                                 'integrate_dataset', 'public_korean_pron', 'check_file_integrity', 'generate_mel_f0', 'cal_f0_scale_per_training_speaker'])
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--num_workers', type=int, default=12)
    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    if args.dataset == 'libri_tts':
        assert(True)
        print("Not implemented")
        # preprocess_libri_tts(args)
    elif args.dataset == 'selvas_multi':
        assert(True)
        print("Not implemented")
        # preprocess_selvas_multi(args)
    elif args.dataset == 'integrate_dataset':
        integrate_dataset(args)
    elif args.dataset == 'selvas_multispeaker_pron':
        preprocess_selvas_multispeaker_pron(args)
    elif args.dataset == 'public_korean_pron':
        preprocess_public_korean_pron(args)
    elif args.dataset == 'check_file_integrity':
        check_for_file_integrity(args)
    elif args.dataset == 'generate_mel_f0':
        gen_mel_f0(args)
    elif args.dataset == 'cal_f0_scale_per_training_speaker':
        preprocess_cal_f0_scale_per_training_speaker(args)


if __name__ == "__main__":
    main()
