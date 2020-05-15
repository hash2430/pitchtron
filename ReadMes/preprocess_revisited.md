# preprocessing steps to run for multi-speaker Korean TTS
```
python preprocess.py --dataset={following keywords}
```
Run them in following order. You can opt out some parts depending on your needs.

1. selvas_multispeaker_pron
2. public_korean_pron
3. integrate_dataset
4. check_file_integrity
5. generate_mel_f0

## 1. selvas_multispeaker_pron
* raw pcm to wav_22050 under each speaker
    * src:'{data_root}/{speaker}/raw/*.pcm'
    * dst: '{data_root}/{speaker}/wav_22050/*.wav'
* trim with 25 top dB
* data split: for every 400 audio, make it eval & the same for test
    * train: 33194 wavs
    * eval: 83 wavs
    * test: 84 wavs
* generate meta file with script that is in phoneme
    *filelists/single_language_selvas/train_file_list_pron.txt
    
## 2. public_korean_pron
* where to download: [서울말 낭독체 발화 말뭉치](https://ithub.korean.go.kr/user/total/referenceManager.do)
* regularize sampling rate to 22050 Hz (This DB has irregular sr) 
* Trim with top 25 dB
* source: 
    * wav_16000/{speaker}/*.wav
    * pron/{speaker}/t**.txt
    * Excluded from script:
        * the script for unzipping and moving the wavs to wav_16000 is not included. You need to make it this form yourself
        * Text file for all speakers are equal in this DB, so I divided this shared script by literature manually.(It includes missing newline errors so I had to do it manually)
        * Also, the script for G2P is also not included
        * Additional errors in this DB are
        ```
        1. Missing speaker:  fy15, mw12
        2. Wrong data format: mw13_t01_s11.wav, mw13_t01_s12.wav, mw02_t10_s08.wav
        3. Overlapping files and naming mistakes: mv11_t07_s4' (==mv11_t07_s40), fy17_t15_s18(==fy17_t16_s01), fv18_t07_s63(==fv18_t07_s62) 
        ```
* dst: wav_22050/{speaker}/*.wav

## 3. integrate_dataset
* I integrate above two Korean DBs.
* This can be generalized to multi-lingual TTS where there are multiple DBs of different languages.
* Thus, language code correspoding to each DB is appended to the integrated meta text file created in this step.
* How to

1. Modify source file lists('train_file_lists', 'eval_file_lists', 'test_file_lists') and target file lists(target_train_file_list, target_eval_file_list, target_test_file_list) 
 from preprocess.preprocess.integrate_dataset(args)
2. You might want to modify _integrate() method to designate **language code** for each DB. Sorry it is hard-codded for now.
3. Run preprocess.py
```
python preprocess.py --dataset=integrate_dataset
```

## 4. check_file_integrity
* This step generates meta file with wav paths that has been unable to read.
* You might wanna remove them from your final filelists or go through some investigation. It's on you. This step does not remove these detected files from the filelists.
* out: problematic_merge_korean_pron_{}.txt 

## 5. generate_mel_f0
* I probably could have integrated this part to step 1 and 2, but I didn't do it at that time and it turned out this slows down the learning process seriously.
* So I recommend users to run this step to speed up the learning process.
* src: wav_22050/*.wav
* dst: mel/\*.pt and f0/\*.pt

## 6. initialize first few epochs with single speaker DB
* Prepare separate train, valid filelists for single speaker 
* Files for single speaker training and validation are also included in multispeaker filelists.
* I experimented training initial 30 epochs with single speaker DB and it helped learning encoder-decoder alignment a lot.
## TODO
* Undo step 5
    * Generating mel and f0 in preprocessing time did not increase learning speed.
    * It seems the file/io and mel/f0 calculating speed is about the same.
    * There is no point preprocessing them. It only takes up disk memory.
    * It also took 1 day to generate them
