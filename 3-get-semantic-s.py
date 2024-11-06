import os

# inp_text = os.environ.get("inp_text")
# exp_name = os.environ.get("exp_name")
# i_part = os.environ.get("i_part")
# all_parts = os.environ.get("all_parts")
# print(f"{inp_text} {exp_name} {i_part} {all_parts}")
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("_CUDA_VISIBLE_DEVICES")
# opt_dir = os.environ.get("opt_dir")
# pretrained_s2G = os.environ.get("pretrained_s2G")
# s2config_path = os.environ.get("s2config_path")
# is_half = eval(os.environ.get("is_half", "True"))
import math, traceback
import multiprocessing
import sys, pdb
import numpy as np

now_dir = os.getcwd()
sys.path.append(now_dir)
from random import shuffle
import torch.multiprocessing as mp
from glob import glob
from tqdm import tqdm
import logging, librosa, utils, torch
from ttv_v1.t2w2v_transformer import SynthesizerTrn
from text.symbols_lmdh import symbols, tone_symbols, language_symbols

logging.getLogger("numba").setLevel(logging.WARNING)
# from config import pretrained_s2G

# inp_text=sys.argv[1]
# exp_name=sys.argv[2]
# i_part=sys.argv[3]
# all_parts=sys.argv[4]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[5]
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name

ckpt_file = "logs/exp12_test/G_20.pth"
wav_dir = "/data2/liuhuang/VCTK-Corpus-Processed/wav48/p376/audio"
hmel_lists = os.listdir(wav_dir)
hmel_lists = [os.path.join(wav_dir, item) for item in hmel_lists if item.endswith("hmel.npy")]

# hubert_dir = "%s/4-cnhubert" % (opt_dir)
# semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)


device = "cuda:0"
hps_t2w2v = utils.get_hparams_from_file(os.path.join(os.path.split(ckpt_file)[0], 'config.json'))
vq_model = SynthesizerTrn(len(symbols),
    len(tone_symbols),
    len(language_symbols),
#   text_aligner, # ,
    hps_t2w2v.data.filter_length // 2 + 1,
    hps_t2w2v.data.hop_length,
    hps_t2w2v.data.sampling_rate,
    hps_t2w2v.train.segment_size // hps_t2w2v.data.hop_length,
    **hps_t2w2v.model).to(device)
print(f"finish init text2w2v")
# text2w2v.load_state_dict(torch.load(a.ckpt_text2w2v))
vq_model, _, _, _ = utils.load_checkpoint(ckpt_file, vq_model, None)
vq_model.eval()

# def name2go(wav_name, lines):
#     hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
#     if os.path.exists(hubert_path) == False:
#         return
#     ssl_content = torch.load(hubert_path, map_location="cpu")
#     if is_half == True:
#         ssl_content = ssl_content.half().to(device)
#     else:
#         ssl_content = ssl_content.to(device)
#     codes = vq_model.extract_latent(ssl_content)
#     semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
#     lines.append("%s\t%s" % (wav_name, semantic))

def name2go1(mel_path):
    # hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
    # if os.path.exists(hubert_path) == False:
    #     return
    # ssl_content = torch.load(hubert_path, map_location="cpu")
    # if is_half == True:
    #     ssl_content = ssl_content.half().to(device)
    # else:
    #     ssl_content = ssl_content.to(device)

    sem_path = mel_path.replace("hmel.npy", "hsem.npy")
    if os.path.exists(sem_path):
        return
    
    ssl_content = np.load(mel_path)
    ssl_content = torch.from_numpy(ssl_content)
    ssl_content = ssl_content.unsqueeze(0).to(device)
    codes = vq_model.extract_latent(ssl_content)
    semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
    print(semantic)

for item in tqdm(hmel_lists):
    name2go1(item)
    exit()

# with open(inp_text, "r", encoding="utf8") as f:
#     lines = f.read().strip("\n").split("\n")

# lines1 = []
# for line in lines[int(i_part) :: int(all_parts)]:
#     # print(line)
#     try:
#         # wav_name,text=line.split("\t")
#         wav_name, spk_name, language, text = line.split("|")
#         wav_name = os.path.basename(wav_name)
#         # name2go(name,lines1)
#         name2go(wav_name, lines1)
#     except:
#         print(line, traceback.format_exc())
# with open(semantic_path, "w", encoding="utf8") as f:
#     f.write("\n".join(lines1))
