import torch
import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read
import numpy as np
import sys



hann_window = {}
mel_basis = {}

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output



def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

def get_audio(filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        audio = audio
        if sampling_rate != 16000:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, 16000))
        audio_norm = audio / 32768.0
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
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
    
def main():
    # target_trans_file = sys.argv[1] #transcription.txt
    target_16k_audio_path = sys.argv[1] #16k音频所在路径
    
    
    # transcription = open(target_trans_file).readlines()
    
    for wav in os.listdir(target_16k_audio_path):
        wav_path = os.path.join(target_16k_audio_path , wav)
        if not os.path.exists(wav_path.replace('.wav','.mel.npy')):
            mel = get_audio(wav_path).cpu()
            np.save(wav_path.replace('.wav','.mel.npy'),mel.numpy())
            # print(wav_path)
            
    
    # while line_index < len(transcription)-2:
    #     chi_line = transcription[line_index]
    #     wav_id = chi_line.strip().split('\t')[0]
    #     wav_path = os.path.join(target_16k_audio_path, wav_id+'.wav')
    #     if os.path.exists(wav_path):
    #         mel = get_audio(wav_path).cpu()
    #         np.save(wav_path.replace('.wav','.mel.npy'),mel.numpy())
    #     line_index = line_index + 2
        
if __name__ == "__main__":
    main()
    

    
    