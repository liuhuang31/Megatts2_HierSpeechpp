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
- 1. s2_stage: use train_ms.py train megatts(rvq).
- 2. s1_stage: use train_ms_s1.py train plm, config.json train_stage param set "s1_1"; s1_stage's exp_dir as s2_stage's to load RVQ related model checkpoint.
```bash
# train s2_stage
# for conv stride 8: in data_utils, dur mel w2v use 8 times
CUDA_VISIBLE_DEVICES="0" python train_ms.py -c configs/config.json -m exp

# train s1_stage: config.json train_stage param set "s1_1".
# train plm GPT, not use GPT-SoVITS's AR modules model;
# and to avoid some risks, we use github_megatts2's GPT model, code implementation is not carefully checked...
CUDA_VISIBLE_DEVICES="0" python train_ms_s1.py -c configs/config.json -m exp
```

## Inference
The provide model checkpoint [Models](https://huggingface.co/liuhuang/Megatts2_HierSpeechpp), use [zhvoice](https://github.com/fighting41love/zhvoice), LibriTTS(100,360,500), VCTK, aishell3 and 200h_chinese(generated from the TTS interface...).
- Download the provided checkpoint to 'models' dir, or change the checkpoint path to your owns.
```bash
python inference_plm.py
```

## More
For HierSpeechpp's vocoder is heavy and not open source training code.<br/> You can use hiftnet as vocoder, it also need f0 to train the model. You can see https://github.com/yl4579/HiFTNet <br/>
Also, if you want to wav/audio super-resolution 16/24 kHz to 48 kHz, go to https://github.com/liuhuang31/HiFTNet-sr
* important: retrain HiFTNet or HiFTNet-sr, its feature need change to wav2vec.
