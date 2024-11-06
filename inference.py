import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import re
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write
import torchaudio
import utils
from Mels_preprocess import MelSpectrogramFixed

from hierspeechpp_speechsynthesizer import (
    SynthesizerTrn
)
from ttv_v1.text import text_to_sequence
from ttv_v1.t2w2v_transformer import SynthesizerTrn as Text2W2V
from speechsr24k.speechsr import SynthesizerTrn as SpeechSR24
from speechsr48k.speechsr import SynthesizerTrn as SpeechSR48
from denoiser.generator import MPNet
from denoiser.infer import denoise
from text.symbols_lmdh import symbols, tone_symbols, language_symbols
from train_ms import load_ASR_models
from data_utils import get_tone
from text import cleaned_text_to_sequence_lmdh, cleaned_tone_to_sequence_lmdh

seed = 1111
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def load_text(fp):
    with open(fp, 'r') as f:
        filelist = [line.strip() for line in f.readlines()]
    return filelist
def load_checkpoint(filepath, device):
    print(filepath)
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict
def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param
def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result

def add_blank_token(text):

    text_norm = intersperse(text, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_text_tone(text):
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
    tone_list = get_tone(text)
    tone = cleaned_tone_to_sequence_lmdh(tone_list)
    text = re.sub(r"([a-zA-Z])\d", r"\1", text)
    text = cleaned_text_to_sequence_lmdh(text)
    return tone, text

def get_text(text):        
    try:
        tone, text = get_text_tone(text)
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

    # language = torch.LongTensor(language)
    return text, tone, language
    
def tts(text, a, hierspeech, prompt_path="", speaker_name="", item_count=1):
    
    net_g, text2w2v, speechsr, denoiser, mel_fn = hierspeech

    os.makedirs(a.output_dir, exist_ok=True)
    output_dir = os.path.join(a.output_dir, speaker_name)
    os.makedirs(output_dir, exist_ok=True)

    # text = text_to_sequence(str(text), ["english_cleaners2"])
    # token = add_blank_token(text).unsqueeze(0).cuda()
    text, tone, language = get_text(text)
    text_length = torch.LongTensor([text.size(-1)]).cuda() 
    text, tone, language = text.unsqueeze(0), tone.unsqueeze(0), language.unsqueeze(0)
    text, tone, language = text.cuda(), tone.cuda(), language.cuda()
    print(f"text: {text.shape}, text_length: {text_length}, tone: {tone.shape}, language: {language.shape}")
    # Prompt load
    # audio, sample_rate = torchaudio.load(a.input_prompt)
    audio, sample_rate = torchaudio.load(prompt_path)

    # support only single channel
    audio = audio[:1,:] 
    # Resampling
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000, resampling_method="kaiser_window") 
    if a.scale_norm == 'prompt':
        prompt_audio_max = torch.max(audio.abs())

    # We utilize a hop size of 320 but denoiser uses a hop size of 400 so we utilize a hop size of 1600
    ori_prompt_len = audio.shape[-1]
    p = (ori_prompt_len // 1600 + 1) * 1600 - ori_prompt_len
    audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data
    src_mel_ttv = mel_fn(audio.cuda())
    src_mel_ttv_length = torch.LongTensor([src_mel_ttv.size(-1)]).to(device)
    print(f"src_mel_ttv: {src_mel_ttv.shape} {src_mel_ttv_length}")

    # file_name = os.path.splitext(os.path.basename(a.input_prompt))[0]

    # If you have a memory issue during denosing the prompt, try to denoise the prompt with cpu before TTS 
    # We will have a plan to replace a memory-efficient denoiser 
    if a.denoise_ratio == 0:
        audio = torch.cat([audio.cuda(), audio.cuda()], dim=0)
    else:
        with torch.no_grad():
            denoised_audio = denoise(audio.squeeze(0).cuda(), denoiser, hps_denoiser)
        audio = torch.cat([audio.cuda(), denoised_audio[:,:audio.shape[-1]]], dim=0)

    
    audio = audio[:,:ori_prompt_len]  # 20231108 We found that large size of padding decreases a performance so we remove the paddings after denosing.

    src_mel = mel_fn(audio.cuda())

    src_length = torch.LongTensor([src_mel.size(2)]).to(device)
    src_length2 = torch.cat([src_length,src_length], dim=0)
    print(f"src_mel: {src_mel.shape} {src_length} {src_length2}")

    ## TTV (Text --> W2V, F0)
    with torch.no_grad():
        w2v_x, pitch = text2w2v.infer(text, text_length, src_mel_ttv, src_mel_ttv_length, tone, language, noise_scale=a.noise_scale_ttv, denoise_ratio=a.denoise_ratio)
        print(f"w2v_x: {w2v_x.shape}, pitch: {pitch.shape}")
        src_length = torch.LongTensor([w2v_x.size(2)]).cuda()  
        
        ## Pitch Clipping
        pitch[pitch<torch.log(torch.tensor([55]).cuda())]  = 0

        ## Hierarchical Speech Synthesizer (W2V, F0 --> 16k Audio)
        converted_audio = \
            net_g.voice_conversion_noise_control(w2v_x, src_length, src_mel, src_length2, pitch, noise_scale=a.noise_scale_vc, denoise_ratio=a.denoise_ratio)
                
        ## SpeechSR (Optional) (16k Audio --> 24k or 48k Audio)
        if a.output_sr == 48000: 
            converted_audio = speechsr(converted_audio)
        elif a.output_sr == 24000:
            converted_audio = speechsr(converted_audio)
        else:
            converted_audio = converted_audio

    converted_audio = converted_audio.squeeze()
    
    if a.scale_norm == 'prompt':
        converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 32767.0 * prompt_audio_max
    else:
        converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 32767.0 * 0.999 

    converted_audio = converted_audio.cpu().numpy().astype('int16')

    # file_name2 = "{}.wav".format(file_name)
    output_file = os.path.join(output_dir, str(item_count).zfill(3)+".wav")
    
    if a.output_sr == 48000:
        write(output_file, 48000, converted_audio)
    elif a.output_sr == 24000:
        write(output_file, 24000, converted_audio)
    else:
        write(output_file, 16000, converted_audio)

def model_load(a):
    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).cuda()
    
    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    net_g.load_state_dict(torch.load(a.ckpt))
    _ = net_g.eval()

    # text_aligner = load_ASR_models(hps_t2w2v.train.ASR_path, hps_t2w2v.train.ASR_config).eval()

    text2w2v = Text2W2V(len(symbols),
      len(tone_symbols),
      len(language_symbols),
    #   text_aligner, # ,
      hps_t2w2v.data.filter_length // 2 + 1,
      hps_t2w2v.data.hop_length,
      hps_t2w2v.data.sampling_rate,
    hps_t2w2v.train.segment_size // hps_t2w2v.data.hop_length,
    **hps_t2w2v.model).cuda()
    print(f"finish init text2w2v")
    # text2w2v.load_state_dict(torch.load(a.ckpt_text2w2v))
    text2w2v, _, _, _ = utils.load_checkpoint(a.ckpt_text2w2v, text2w2v, None)
    text2w2v.eval()

    if a.output_sr == 48000:
        speechsr = SpeechSR48(h_sr48.data.n_mel_channels,
            h_sr48.train.segment_size // h_sr48.data.hop_length,
            **h_sr48.model).cuda()
        utils.load_checkpoint(a.ckpt_sr48, audiosr, None)
        audiosr.eval()
       
    elif a.output_sr == 24000:
        speechsr = SpeechSR24(h_sr.data.n_mel_channels,
        h_sr.train.segment_size // h_sr.data.hop_length,
        **h_sr.model).cuda()
        utils.load_checkpoint(a.ckpt_sr, audiosr, None)
        audiosr.eval()
      
    else:
        audiosr = None
    
    denoiser = MPNet(hps_denoiser).cuda()
    state_dict = load_checkpoint(a.denoiser_ckpt, device)
    denoiser.load_state_dict(state_dict['generator'])
    denoiser.eval()
    return net_g, text2w2v, audiosr, denoiser, mel_fn

def inference(a):
    
    hierspeech = model_load(a) 
    # Input Text 
    # exit()
    text = load_text(a.input_txt)
    text = [item.split("|")[2] for item in text if item.strip() != ""]
    prompt = load_text(a.input_prompt)
    prompt = [item.split("|")[0] for item in prompt if item.strip() != ""]
    prompt = [item for item in prompt if item.endswith(".wav")]
    
    
    for item in prompt:
        if not os.path.exists(item):
            print(f"{item} not exists, skip it!")
            continue
        speaker_name = item.split('/')[-3]
        print(f"{speaker_name} gen...")

        item_count = 1
        for tt in text:
            tts(tt, a, hierspeech, prompt_path=item, speaker_name=speaker_name, item_count=item_count)
            item_count += 1

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_prompt', default='./filelists/prompt.txt',help="a text file contains prompt wav dir")
    parser.add_argument('--input_txt', default='./filelists/reference_text.txt', help="a text file contains text")
    parser.add_argument('--output_dir', default='output/exp11_114k')
    parser.add_argument('--ckpt', default='./logs/hierspeechpp_eng_kor/hierspeechpp_v1.1_ckpt.pth')
    parser.add_argument('--ckpt_text2w2v', '-ct', help='text2w2v checkpoint path', default='./logs/exp11/G_114000.pth')
    parser.add_argument('--ckpt_sr', type=str, default='./speechsr24k/G_340000.pth')  
    parser.add_argument('--ckpt_sr48', type=str, default='./speechsr48k/G_100000.pth')  
    parser.add_argument('--denoiser_ckpt', type=str, default='denoiser/g_best')
    parser.add_argument('--scale_norm', type=str, default='max')
    parser.add_argument('--output_sr', type=float, default=16000)
    parser.add_argument('--noise_scale_ttv', type=float,
                        default=0.333)
    parser.add_argument('--noise_scale_vc', type=float,
                        default=0.333)
    parser.add_argument('--denoise_ratio', type=float,
                        default=0.8)
    a = parser.parse_args()

    global device, hps, hps_t2w2v,h_sr,h_sr48, hps_denoiser
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hps = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt)[0], 'config.json'))
    hps_t2w2v = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_text2w2v)[0], 'config.json'))
    h_sr = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_sr)[0], 'config.json') )
    h_sr48 = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_sr48)[0], 'config.json') )
    hps_denoiser = utils.get_hparams_from_file(os.path.join(os.path.split(a.denoiser_ckpt)[0], 'config.json'))

    inference(a)

if __name__ == '__main__':
    main()

'''
CUDA_VISIBLE_DEVICES=1 python inference.py \
                --ckpt "logs/hierspeechpp_eng_kor/hierspeechpp_v1.1_ckpt.pth" \
                --ckpt_text2w2v "logs/ttv_libritts_v1/ttv_lt960_ckpt.pth" \
                --output_dir "tts_results_eng_kor_v2" \
                --noise_scale_vc "0.333" \
                --noise_scale_ttv "0.333" \
                --denoise_ratio "0"

CUDA_VISIBLE_DEVICES=0 python inference.py --input_txt ./filelists/reference_text_en.txt
'''