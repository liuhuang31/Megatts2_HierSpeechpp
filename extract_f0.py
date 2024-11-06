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

torch.set_num_threads(1)

def get_yaapt_f0(audio, rate=16000, interp=False):
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0, 'f0_max':55, 'f0_max':1100})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]] 
    f0 = np.vstack(f0s)
    return f0


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


def extract_f0(param):
    item = param
    f0_path = item.replace('.wav','.hf0.npy')
    if os.path.exists(f0_path):
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
        f0 = get_yaapt_f0(source_audio.numpy()).squeeze(0).squeeze(0)
    except Exception as e:
        print(f"{item} {e}")
        return
        # raise ValueError(f"{item} {e}")
        f0 = np.zeros((1, 1, source_audio.shape[-1] // 80))
        f0 = f0.astype(np.float32)
        f0 = f0.squeeze(0)

    np.save(f0_path, f0)

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
        for _ in tqdm(pool.imap_unordered(extract_f0,wav_lists),total=len(wav_lists)):
            pass
        print()
    else:
        for item in tqdm(wav_lists):
            extract_f0(item)
    
  
if __name__ == '__main__':
    __cmd()

'''
python extract_f0.py --input_dir /data2/liuhuang/zhvoice/ --mt 36
python extract_f0.py --input_dir /data2/liuhuang/dataset/LibriTTS/ --mt 36
python extract_f0.py --input_dir /data2/liuhuang/dataset/LibriTTS/LibriTTS360/ --mt 12
python extract_f0.py --input_dir /data2/liuhuang/dataset/LibriTTS/LibriTTS500/ --mt 12

CUDA_VISIBLE_DEVICES="1" python extract_f0.py --input_dir /data2/liuhuang/dataset/LibriTTS/LibriTTS500/
'''