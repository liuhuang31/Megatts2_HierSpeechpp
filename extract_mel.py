import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torchaudio
import librosa
import torch
from tqdm import tqdm
import argparse
from torch.nn import functional as F
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import multiprocessing
from scipy.io.wavfile import read
# print(source_audio.shape, source_audio[0:20])
# b, sr = librosa.load(wavpath, sr=16000)
# print(b.shape, b[0:20])
from Mels_preprocess import MelSpectrogramFixed
torch.set_num_threads(1)


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


def find_all_wav_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            if not os.path.splitext(filename)[-1] == '.wav':
                # print('the file:{} is not a wav file,skip it!'.format(filename))
                continue
            apath = os.path.join(maindir, filename) # merge into a full path
            result.append(apath)
    return result


def extract_mel(param):
    item = param
    mel_path = item.replace('.wav','.hmel.npy')
    if os.path.exists(mel_path):
        return

    # use torchaudio 
    try:
        source_audio, sample_rate = torchaudio.load(item)
        if sample_rate != 16000:
            source_audio = torchaudio.functional.resample(source_audio, sample_rate, 16000, resampling_method="kaiser_window")
        p = (source_audio.shape[-1] // 1280 + 1) * 1280 - source_audio.shape[-1]
        source_audio = torch.nn.functional.pad(source_audio, (0, p), mode='constant').data
    except Exception as e:
        print(f"{item} {e}")
        return
        
    try:
        mel = mel_fn(source_audio.cuda()).squeeze(0)
    except Exception as e:
        print(f"{item} {e}")
        return
    np.save(mel_path, mel.cpu().numpy())

def __cmd():
    description = "extract f0"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--input_dir",
        type=str,
        default='',
        required=False,
        help="the audio corpus dir.")
    parser.add_argument(
        "--mt",
        type=int,
        default=1,
        help="how much proceess in parallel.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        raise ValueError(f"input_dir not exists: {input_dir}")
    
    wav_lists = find_all_wav_path(input_dir)
    if args.mt:
        print("using multiprocessing...")
        pool = multiprocessing.Pool(int(args.mt))
        for _ in tqdm(pool.imap_unordered(extract_mel,wav_lists),total=len(wav_lists)):
            pass
        print()
    else:
        for item in tqdm(wav_lists):
            extract_mel(item)
    
  
if __name__ == '__main__':
    __cmd()

'''
python extract_f0.py --input_dir /data2/liuhuang/zhvoice/ --mt 0
python extract_mel.py --input_dir /data2/liuhuang/dataset/LibriTTS/ --mt 32

# use this
CUDA_VISIBLE_DEVICES="0" python extract_mel.py --input_dir /data2/liuhuang/dataset/LibriTTS/ --mt 0
CUDA_VISIBLE_DEVICES="0" python extract_mel.py --input_dir /data2/liuhuang/zhvoice/  --mt 0
'''