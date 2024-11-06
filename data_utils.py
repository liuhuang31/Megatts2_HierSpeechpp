import time
import os
import re
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio
from torch.nn import functional as F
from Mels_preprocess import MelSpectrogramFixed

import commons 
import librosa
from mel_processing import spectrogram_torch
# from utils import load_wav_to_torch, load_filepaths_and_text
# from ttv_v1.text import text_to_sequence, cleaned_text_to_sequence, rhy_to_sequence
from ttv_v1.t2w2v_transformer import Wav2vec2
from mel_processing import spec_to_mel_torch
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
from text import cleaned_text_to_sequence_lmdh, cleaned_tone_to_sequence_lmdh
from text.symbols_lmdh import _punc


def get_wav_dur(full_path):
  duration = librosa.get_duration(filename=full_path)
  return duration


def get_data_path_list(train_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    return train_list


# def load_filepaths_and_text_lmdh(filepaths_and_text_list, split="|"):
# #   with open(filename, encoding='utf-8') as f:
# #     filepaths_and_text_list = [line.strip() for line in f if line.strip()!=""]
#   filepaths_and_text_all = []
#   for filepath in filepaths_and_text_list:
#     if not os.path.exists(filepath):
#       raise ValueError(f"{filepath} not exists!")
#     with open(filepath, encoding='utf-8') as f:
#       filepaths_and_text = [line.strip().split(split) for line in f if line.strip()!=""]
#       filepaths_and_text_all += filepaths_and_text

#   return filepaths_and_text_all


def load_filepaths_and_text_lmdh(filepaths_and_text_list, split="|"):
#   with open(filename, encoding='utf-8') as f:
#     filepaths_and_text_list = [line.strip() for line in f if line.strip()!=""]
  filepaths_and_text_all = []
  for filepath in filepaths_and_text_list:
    if not os.path.exists(filepath):
        raise ValueError(f"{filepath} not exists!")
    with open(filepath, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f if line.strip()!=""]
        # print(f"filepaths_and_text: {filepaths_and_text[0]}")
        ## make mrte's mel
        lens = len(filepaths_and_text)
        for i in range(lens):
            if i+1 < lens:
                first_audio = filepaths_and_text[i+1][0]
            else:
                first_audio = filepaths_and_text[i][0]
            if i+2 < lens:
                second_audio = filepaths_and_text[i+2][0]
            else:
                second_audio = filepaths_and_text[i][0]
            if os.path.exists(first_audio.replace('.wav','.hmel.npy')) and os.path.exists(second_audio.replace('.wav','.hmel.npy')):
                filepaths_and_text[i].append(first_audio + "+" + second_audio)
            else:
                filepaths_and_text[i].append(filepaths_and_text[i][0])
        filepaths_and_text_all += filepaths_and_text

  return filepaths_and_text_all


mel_fn = MelSpectrogramFixed(
    sample_rate=16000,
    n_fft=1280,
    win_length=1280,
    hop_length=320,
    f_min=0,
    f_max=8000,
    n_mels=80,
    window_fn=torch.hann_window
).cuda()


def get_mel(filename):
    audio, sampling_rate = torchaudio.load(filename)
    if sampling_rate != 16000:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, 16000))
    audio_norm = audio / 32768.0
    audio_norm = audio_norm.cuda()
    spec_filename = filename.replace(".wav", ".spec.pt")
    if os.path.exists(spec_filename):
        spec = torch.load(spec_filename)
    else:
        # print("audio_norm.shape____:",audio_norm.shape)
        spec = spectrogram_torch(audio_norm, 1280,
            16000, 320, 1280,
            center=False)
        # spec = torch.squeeze(spec, 0)
    #     torch.save(spec, spec_filename)
    # return spec, audio_norm
    mel = spec_to_mel_torch(
        spec, 
        1280, 
        80, 
        16000,
        0, 
        None)
    return mel.squeeze(0)


def get_yaapt_f0(audio, rate=16000, interp=False):
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
    # for y in audio.double():
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        # y_pad = F.pad(y.squeeze(), (to_pad, to_pad), "constant", value=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0, 'f0_max':1100})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]] 
    f0 = np.vstack(f0s)
    return f0
    

def is_english_phoneme_for_spss(text_string):
    """判断一个传入字符是否是英文phoneme"""
    if re.search(r'^[A-Z]', text_string):
        return True
    else:
        return False

def is_number(text_string):
    """判断一个传入字符是否是数字"""
    if re.search(r'^\d', text_string):
        return True
    else:
        return False  
    

def get_tone(text):
    split_text = [item.strip() for item in text.split() if item.strip() !=""]
    tone_list = []
    pre_tone = "0" # tone of the previous phoneme 
    len_split_text = len(split_text)
    for i in range(len_split_text-1,-1,-1):
        item = split_text[i]
        if item in _punc or item == "<blank>" or item == "#2":
            tone = "0"
        elif is_english_phoneme_for_spss(item):
            if is_number(item[-1]):
                tone = str(int(item[-1])+7)
            else:
                tone = "6"
        else:
            if is_number(item[-1]):
                tone = item[-1]
                pre_tone = tone
            else:
                tone = pre_tone
        tone_list.append(tone)
    # reverse data
    tone_list = tone_list[::-1]
    # for start sil
    tone_list[0] = '0'
    return tone_list


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        # self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 800)
        self.max_wav_len = getattr(hparams, "max_wav_len", 18.0)
        self.min_wav_len = getattr(hparams, "min_wav_len", 1.00)
        self.max_mel_mrte_len = getattr(hparams, "max_mel_mrte_len", 1200) # 24s  

        data_list = get_data_path_list(audiopaths_sid_text)
        data_list = [item.strip() for item in data_list if item.strip() != ""]
        _data_list = load_filepaths_and_text_lmdh(data_list)
        print(f"_data_list: {_data_list[0]}")
        _data_list = [[d[0], d[2], d[1], d[3]] for d in _data_list if os.path.exists(d[0])]
        self.audiopaths_sid_text = [data if len(data) == 4 else (*data, 0) for data in _data_list]

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        print(f"self.audiopaths_sid_text: {self.audiopaths_sid_text[0]}")
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        max_text_len = 0
        max_wav_len = 0.0
        min_wav_len = 20
        audiopaths_sid_text_new = []
        print(f"before filter, origin data nums: {len(self.audiopaths_sid_text)}")
        lengths = []
        for audiopath, text, _, mel_cat_list in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len and os.path.exists(audiopath) and os.path.exists(audiopath.replace('.wav','.hw2v.pt')) and os.path.exists(audiopath.replace('.wav','.hf0.npy')) and os.path.exists(audiopath.replace('.wav','.hmel.npy')) and os.path.exists(audiopath.replace('.wav', '.dur.npy')):
                # duration = get_wav_dur(audiopath)
                duration = os.path.getsize(audiopath) / 2.0 / 16000.0 # for all wav is 16k

                if "zhvoice" in audiopath:
                    dur = np.load(audiopath.replace('.wav','.dur.npy'))
                    dur = map(np.float, dur[:, -1])
                    dur = np.array(list(dur))
                    dur_length = sum(dur)
                    if duration - dur_length > 0.04 or dur_length - duration > 0.04:
                        print(f"{audiopath} duration:{duration} and mfa_dur:{dur_length} not align!")
                        continue

                    w2v = torch.load(audiopath.replace('.wav','.hw2v.pt'))
                    dur_length = np.round(np.divide(dur_length, 0.01))
                    w2v_length = w2v.shape[-1]
                    if w2v_length*2 - dur_length > 6 or dur_length - w2v_length*2 > 6:
                        print(f"{audiopath} w2v_length:{w2v_length*2} and mfa_dur:{dur_length} not align!")
                        continue

                # tone, text = self.get_text_tone(text)
                # if len(tone) != dur.shape[0]:
                #     print(f"{audiopath} tone:{len(tone)} and mfa_dur_length:{dur.shape[0]} not align!")
                #     continue
                
                if duration > max_wav_len:
                    max_wav_len = duration
                    print("max", duration, audiopath)
                if duration < min_wav_len:
                    min_wav_len = duration
                    print("min", duration, audiopath)
                if duration > self.max_wav_len or duration < self.min_wav_len:
                    continue
                if len(text) >= (int(duration/0.01)//2):
                    # print(f"text_len({len(text)}) >= ((duration/0.01)//2)({int(duration/0.01)//2}): {audiopath}")
                    continue

                audiopaths_sid_text_new.append([audiopath, text, mel_cat_list])
                # lengths.append(int(duration // (self.hop_length/self.sampling_rate)))
                # lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
                lengths.append(os.path.getsize(audiopath) // (2 * 320))
            if len(text) > max_text_len:
                max_text_len = len(text)
                
        print(f"before filter, origin data nums: {len(self.audiopaths_sid_text)}")
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        print(f"after filter, origin data nums: {len(self.audiopaths_sid_text)}")
        print(f"max text length: {max_text_len}")
        print(f"min wav time: {min_wav_len}s")
        print(f"max wav time: {max_wav_len}s")
        self.get_audio_text_speaker_pair(self.audiopaths_sid_text[0])

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, text, mel_cat_list = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2] #加入韵律标签序列 
        text, tone, language = self.get_text(text)
        w2v, mel, mel_mrte, pitch, dur = self.get_w2v(audiopath, mel_cat_list)
        # print(f"tone: {tone.shape} dur: {dur.shape}")
        # print(f"w2v: {w2v.shape}, mel: {mel.shape}, dur: {sum(dur)} pitch: {pitch.shape}")

        if tone.shape[0] != dur.shape[0]:
            print(f"{audiopath} tone:{tone.shape[0]} {dur.shape[0]} not align!")
        ## for llm conv pooling stride set 8
        padded = w2v.shape[-1] % 8
        if padded > 0:
            # mel = torch.cat((mel, mel[:, -1-padded:-1]), dim=-1)
            w2v = torch.cat((w2v, w2v[:, -1-padded:-1]), dim=-1)
            # pitch = torch.cat((pitch, pitch[-1-padded*4:-1]))
            dur[-1] += padded*2
        ## mel and pitch align w2v
        # print(f"middle padded: {padded}, w2v: {w2v.shape}, mel: {mel.shape}, dur: {sum(dur)} pitch: {pitch.shape}")
        w2v_length = w2v.shape[-1]
        mel_padded = w2v_length - mel.shape[-1]
        pitch_padded = w2v_length*4 - pitch.shape[-1]
        if mel_padded > 0:
            mel = torch.cat((mel, mel[:, -1-mel_padded:-1]), dim=-1)
        else:
            mel = mel[:,:w2v_length]
        if pitch_padded > 0:
            pitch = torch.cat((pitch, pitch[-1-pitch_padded:-1]), dim=-1)
        else:
            pitch = pitch[:w2v_length*4]
        # print(f"after padded: {padded}, w2v: {w2v.shape}, mel: {mel.shape}, dur: {sum(dur)} pitch: {pitch.shape}") 
        return (text, mel, w2v, pitch, tone, language, dur, audiopath, mel_mrte)
    
    def get_w2v(self,audiopath, mel_cat_list):
        if not os.path.exists(audiopath.replace('.wav','.hw2v.pt')) or not os.path.exists(audiopath.replace('.wav','.hf0.npy')) or not os.path.exists(audiopath.replace('.wav','.hmel.npy')):
            source_audio, sample_rate = torchaudio.load(audiopath)
            if sample_rate != 16000:
                source_audio = torchaudio.functional.resample(source_audio, sample_rate, 16000, resampling_method="kaiser_window")  # sinc_interpolation  kaiser_window
            p = (source_audio.shape[-1] // 1280 + 1) * 1280 - source_audio.shape[-1]
            source_audio = torch.nn.functional.pad(source_audio, (0, p), mode='constant').data

        if os.path.exists(audiopath.replace('.wav','.hw2v.pt')):
            w2v = torch.load(audiopath.replace('.wav','.hw2v.pt')).squeeze(0)
        else:
            y_pad = F.pad(source_audio, (40, 40), "reflect")
            Wav2vec = Wav2vec2(layer=7).cuda()
            wav_vector = Wav2vec(y_pad.cuda())
            w2v = wav_vector.squeeze(0)
            torch.save(wav_vector.cpu(), audiopath.replace('.wav','.hw2v.pt'))
            
        if os.path.exists(audiopath.replace('.wav','.hmel.npy')):  
              mel = np.load(audiopath.replace('.wav','.hmel.npy'))
        else:
            # mel = self.get_mel(audiopath).cpu().numpy()
            mel = mel_fn(source_audio.cuda()).squeeze(0).cpu().numpy()
            np.save(audiopath.replace('.wav','.hmel.npy'),mel)
            
        if os.path.exists(audiopath.replace('.wav','.hf0.npy')):
            pitch = np.load(audiopath.replace('.wav','.hf0.npy'))
        else:
            pitch = get_yaapt_f0(source_audio.numpy()).squeeze(0).squeeze(0)
            np.save(audiopath.replace('.wav', ".hf0.npy"), pitch)

        mel_mrte = mel
        # print(f"mel_mrte: {mel_mrte.shape}")
        for mel_cat in mel_cat_list.split('+'):
            mel_mrte = np.concatenate((mel_mrte, np.load(mel_cat.replace('.wav','.hmel.npy'))), axis=1)
        # max length limit
        if mel_mrte.shape[-1] > self.max_mel_mrte_len:
            mel_mrte = mel_mrte[:, :self.max_mel_mrte_len]
        # random slice mel_mrte
        start_idx = random.randint(0, mel_mrte.shape[-1] // 2)
        mel_mrte = mel_mrte[:, start_idx:start_idx + mel_mrte.shape[-1] // 2]
        # print(f"mel: {mel.shape}, mel_mrte: {mel_mrte.shape}")
        ## process durration
        dur = np.load(audiopath.replace('.wav','.dur.npy'))
        dur = map(np.float, dur[:, -1])
        dur = np.array(list(dur))
        dur = np.round(np.divide(dur, 0.01))
        # print(f"dur: {dur}")

        w2v_length = w2v.shape[-1]
        dur_length = int(sum(dur))
        if w2v_length*2 - dur_length > 10 or dur_length - w2v_length*2 > 10:
            print(f"{audiopath} dur:{dur_length} and w2v:{w2v_length*2} not align!")
        # print(f"dur_length: {dur_length}, w2v_length: {w2v_length*2}")
        if w2v_length*2 - dur_length > 0:
            begin = (w2v_length*2 - dur_length) // 2
            end = w2v_length*2 - dur_length - begin
            dur[0] += begin
            dur[-1] += end
        if dur_length - w2v_length*2 > 0:
            dur[-1] -= dur_length - w2v_length*2
        return w2v, torch.from_numpy(mel), torch.from_numpy(mel_mrte), torch.from_numpy(pitch), torch.from_numpy(dur)

    def get_text(self, text):  
        # print(f"text: {text}")      
        try:
            tone, text = self.get_text_tone(text)
        except Exception as e:
            print(e)
            print(text)

        # tone.insert(0, 0)
        # tone.append(0)
        # text.insert(0, 0)
        # text.append(0)
        text = torch.LongTensor(text)
        tone = torch.LongTensor(tone)

        language = text
        condition0 = language == 0 # 
        condition1 = language > 0 # 
        condition2 = language < 74 # 74 'AA'
        condition3 = language >= 74 # 74 'AA'
        condition4 = language < 113 # 113 '?'
        language = torch.where(condition0, 0, language)
        language = torch.where(condition1 & condition2, 1, language)
        language = torch.where(condition3 & condition4, 2, language)
        language = torch.where(language >= 113, 0, language)
        # print(f"text: {text}")
        # print(f"tone: {tone}")
        # print(f"language: {language}")
        # language = torch.LongTensor(language)
        return text, tone, language
    
    def get_text_tone(self, text):
        """for the origin code not add blank between phoneme, we also not add.
            #0|#1 remove it, not as a phoneme.
            #2 its phoneme is "#2".
            #3_punct, remove #3, keep punct as phoneme.
            #4_punct, remove #4, keep punct as phoneme.
        """
        text = re.sub(r'#0|#1|#3|#4', r"", text)
        # remove eos
        text = re.sub(r"eos", r"", text)
        text = re.sub(r"\s+", r" ", text).strip()
        # print(f"text: {len(text)} {text.split()}")
        tone_list = get_tone(text)
        tone = cleaned_tone_to_sequence_lmdh(tone_list)
        text = re.sub(r"([a-zA-Z])\d", r"\1", text)
        text = cleaned_text_to_sequence_lmdh(text)
        return tone, text
        
    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: (text, mel, w2v, sid, phone_dur, pitch, rhy, language_id)
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[2].size(1) for x in batch]),
            dim=0, descending=True)   #按照w2v排序
        
        # (text, mel, w2v, pitch, tone, language)
        max_text_len = max([len(x[0]) for x in batch])
        # max_spec_len = max([x[1].size(1) for x in batch])
        max_mel_len = max([x[1].shape[1] for x in batch])  #mel用来提取spk信息
        # max_wav_len = max([x[2].size(1) for x in batch])
        max_w2v_len = max([x[2].size(1) for x in batch]) #w2v
        max_pitch_len = max([len(x[3]) for x in batch])
        max_mel_mrte_len = max([x[8].shape[1] for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        mel_lengths = torch.LongTensor(len(batch))
        mel_mrte_lengths = torch.LongTensor(len(batch))
        # wav_lengths = torch.LongTensor(len(batch))
        w2v_lengths = torch.LongTensor(len(batch))
        pitch_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        mel_padded = torch.FloatTensor(len(batch), batch[0][1].shape[0], max_mel_len)
        w2v_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_w2v_len)
        pitch_padded = torch.LongTensor(len(batch), max_pitch_len)  #添加
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)  #添加language id pad
        dur_padded = torch.LongTensor(len(batch), max_text_len)
        paths = ['' for _ in range(len(batch))]
        mel_mrte_padded = torch.FloatTensor(len(batch), batch[0][8].shape[0], max_mel_mrte_len)

        text_padded.zero_()
        mel_padded.zero_()
        w2v_padded.zero_()
        pitch_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        dur_padded.zero_()
        mel_mrte_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            mel = row[1]
            mel_padded[i, :, :mel.shape[1]] = mel
            mel_lengths[i] = mel.shape[1]

            w2v = row[2]
            w2v_padded[i, :, :w2v.size(1)] = w2v
            w2v_lengths[i] = w2v.size(1)

            pitch = row[3]
            pitch_padded[i, :pitch.size(0)] = pitch #添加
            pitch_lengths[i] = pitch.size(0)
            
            tone = row[4]
            tone_padded[i, :tone.size(0)] = tone
            
            language = row[5]
            language_padded[i, :language.size(0)] = language

            dur = row[6]
            dur_padded[i, :dur.size(0)] = dur

            path = row[7]
            paths[i] = path

            mel_mrte = row[8]
            mel_mrte_padded[i, :, :mel_mrte.shape[1]] = mel_mrte
            mel_mrte_lengths[i] = mel_mrte.shape[1]

        if self.return_ids:
            return text_padded, text_lengths, mel_padded, mel_lengths, w2v_padded, w2v_lengths, ids_sorted_decreasing
        # print(f"pitch_lengths: {len(pitch_lengths)} {pitch_padded.shape}")
        return text_padded, text_lengths, mel_padded, mel_lengths, w2v_padded, w2v_lengths, pitch_padded, pitch_lengths, tone_padded, language_padded, dur_padded, paths, mel_mrte_padded, mel_mrte_lengths


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        for i, (a,b) in enumerate(zip(buckets, num_samples_per_bucket)):
            if b == 0:
                del buckets[i]
                del num_samples_per_bucket[i]
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
