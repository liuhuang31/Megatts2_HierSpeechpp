# Megatts2_HierSpeechpp
- This project need download some model resoucres and prepare train datasets... 
- The train pipeline needs to be linked together by yourself, and code not clean...
- To avoid some risks, some code implementations changed and not have been carefully checked, may have problems...

* Acoustic model use megatts2
* Vocoder use HierSpeechpp: https://github.com/sh-lee-prml/HierSpeechpp
* Features use facebook's wav2vec2 model extract wav2vec
* For train process: (text, f0, spk_mel, wav2vec) -> megatts2 ->> (wav2vec, f0) -> HierSpeechpp ->> wav
* For inference process: (text, spk_mel) -> megatts2 ->> (wav2vec, f0) -> HierSpeechpp ->> wav

## Model resource 
- Download HierSpeechpp's hierspeechpp_eng_kor, hierspeechpp_libritts960 and ttv_libritts_v1 to this dir.
- Download facebook/wav2vec2-xls-r-300m.

## Dataset prepare
* Features
  * text: use pinyin's phoneme tone(English use CMU).
  * f0: extract_f0.py 
  * mel: extract_mel.py
  * wav2vec: extract_w2v.py
  * duration: phoneme's align duration(use mfa to extract).
* Features filelists
  * text features files
    * configs/config.json need train_list.txt,in the train_list.txt, maybe your need zhvoice/zhmagicdata/5_2431/trans/transcription.txt.styletts.train such files.
  * audio features files
    * in the audio dir, you need also have mel wav2vec f0  dur files.
    * demo.wav
    * demo.hw2v.pt
    * demo.hf0.npy
    * demo.hmel.npy
    * demo.dur.npy

## Train
```bash
# base exp_old, filter data; MaxPool1d.
## config.json train_stage param: "s2" train rvq; "s1" train plm; "s1_1" train plm1;
# for conv stride 8: in data_utils, dur mel w2v use 8 times
CUDA_VISIBLE_DEVICES="2,3" python train_ms.py -c configs/config.json -m exp

# exp_current train_s1, train GPT, not use GPT-SoVITS's AR modules model;
# and to avoid some risks, we use github_megatts2's GPT model, code implementation is not carefully checked...
CUDA_VISIBLE_DEVICES="2,3" python train_ms_s1.py -c configs/config.json -m exp
```