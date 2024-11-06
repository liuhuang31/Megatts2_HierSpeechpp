import os
import json
import argparse
import itertools
import math
import copy
import yaml
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Mels_preprocess import MelSpectrogramFixed
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from Utils.ASR.models import ASRCNN
import logging
import coloredlogs
coloredlogs.install(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=logging.INFO)

import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)
# from models import (
#   SynthesizerTrn,
#   MultiPeriodDiscriminator,
# )

from ttv_v1.t2w2v_transformer import SynthesizerTrn, Megatts2PLM, Megatts2PLM1
from ttv_v1.msd import (
  MultiResSpecDiscriminator,
)

from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
# from ttv_v1.text.symbols import symbols, rhy_symbols, language_id
from text.symbols_lmdh import symbols, tone_symbols, language_symbols
from utils import length_to_mask, maximum_path
from monotonic_align import mask_from_lens

torch.backends.cudnn.benchmark = True
global_step = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    # load ASR model
    def _load_config(path):
        with open(path) as f:
            config = yaml.safe_load(f)
        model_config = config['model_params']
        return model_config

    def _load_model(model_config, model_path):
        model = ASRCNN(**model_config)
        params = torch.load(model_path, map_location='cpu')['model']
        model.load_state_dict(params)
        return model

    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model = _load_model(asr_model_config, ASR_MODEL_PATH)
    _ = asr_model.train()

    return asr_model

def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '53913'

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  import logging
  import coloredlogs
  coloredlogs.install(
          fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
          level=logging.INFO)
  logging.info(hps)

  if rank == 0:
    # logger = utils.get_logger(hps.model_dir)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)
  
  #帧移变成了20ms，这里改了一下boundaries
  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)   #(text, mel, w2v, sid,phone_dur,pitch, rhy, language_id)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,150,250,300,400,500,600,700,800,900],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate()
  #text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid,dur_padded,pitch_padded, rhy_padded
  # collate_fn(train_dataset)
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
  print(f"train_loader: {train_sampler.num_samples} {train_sampler.__len__()}")
  # load pretrained ASR model
  # text_aligner = load_ASR_models(hps.train.ASR_path, hps.train.ASR_config)
  # text_aligner = text_aligner.eval().cuda(rank)
  text_aligner = None
  # text_aligner_copy = copy.deepcopy(text_aligner)
  train_stage = hps.train.train_stage
  logging.info(f"train_stage: {train_stage}")

  net_g = SynthesizerTrn(
      len(symbols),
      len(tone_symbols),
      len(language_symbols),
      # text_aligner,
      hps.data.filter_length // 2 + 1,
      hps.data.hop_length,
      hps.data.sampling_rate,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  net_d = MultiResSpecDiscriminator().cuda(rank)
  # net_g.load_state_dict(torch.load("logs/Aria_Hier_v3/G_80000.pth")["model"])
  
  net_plm = Megatts2PLM().cuda(rank)
  net_plm1 = Megatts2PLM1().cuda(rank)

  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate*0.25, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_p = torch.optim.AdamW(
      net_plm.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_p1 = torch.optim.AdamW(
      net_plm1.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  
  net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
  net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
  net_plm = DDP(net_plm, device_ids=[rank], find_unused_parameters=True)
  net_plm1 = DDP(net_plm1, device_ids=[rank], find_unused_parameters=True)

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
    global_step = int(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth").split('_')[-1].split('.')[0])
  except:
    epoch_str = 1
    global_step = 0
  
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "P_*.pth"), net_plm, optim_p)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "P1_*.pth"), net_plm1, optim_p1)
  except:
    plm_epoch_str = 1
    plm_global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_p = torch.optim.lr_scheduler.ExponentialLR(optim_p, gamma=hps.train.lr_decay, last_epoch=plm_epoch_str-2)
  scheduler_p1 = torch.optim.lr_scheduler.ExponentialLR(optim_p1, gamma=hps.train.lr_decay, last_epoch=plm_epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d, net_plm, net_plm1], [optim_g, optim_d,optim_p,optim_p1], [scheduler_g, scheduler_d,scheduler_p,scheduler_p1], scaler, [train_loader, eval_loader], logging, [writer, writer_eval], text_aligner, train_stage)
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d, net_plm, net_plm1], [optim_g, optim_d,optim_p,optim_p1], [scheduler_g, scheduler_d,scheduler_p,scheduler_p1], scaler, [train_loader, None], None, None, text_aligner, train_stage)
    scheduler_g.step()
    # scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logging, writers, text_aligner, train_stage):
  net_g, net_d, net_plm, net_plm1 = nets
  optim_g, optim_d,optim_p,optim_p1 = optims
  scheduler_g, scheduler_d,scheduler_p,scheduler_p1 = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step
  
  if train_stage == "s2":
    net_g.train()
    net_d.train()
  else:
    net_g.eval()
    net_d.eval()
    net_plm.train()
    net_plm1.train()
  #text_padded, text_lengths, mel_padded, mel_lengths, w2v_padded, w2v_lengths, sid,dur_padded,pitch_padded, rhy_padded, language_id_padded
  # text_padded, text_lengths, mel_padded, mel_lengths, w2v_padded, w2v_lengths, pitch_padded, pitch_lengths, tone_padded, language_padded
  for batch_idx, (x, x_lengths, mel_spk, mel_spk_lengths, w2v, w2v_lengths, pitch,pitch_lengths,tone,language, dur, paths, mrte_mel, mrte_mel_lengths) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    mel_spk, mel_spk_lengths = mel_spk.cuda(rank, non_blocking=True), mel_spk_lengths.cuda(rank, non_blocking=True)
    w2v, w2v_lengths = w2v.cuda(rank, non_blocking=True), w2v_lengths.cuda(rank, non_blocking=True)
    pitch, pitch_lengths = pitch.cuda(rank, non_blocking=True), pitch_lengths.cuda(rank, non_blocking=True)  #添加
    tone = tone.cuda(rank, non_blocking=True)
    language = language.cuda(rank, non_blocking=True)
    dur = dur.cuda(rank, non_blocking=True)
    mrte_mel, mrte_mel_lengths = mrte_mel.cuda(rank, non_blocking=True), mrte_mel_lengths.cuda(rank, non_blocking=True)

    ################## s1 train ##################
    with autocast(enabled=hps.train.fp16_run):
        # get LR features
        with torch.no_grad():
          x_frame, lr_codes = net_g.module.extract_tc_latent_code(x, x_lengths, mel_spk, mel_spk_lengths, tone, language,dur, mrte_mel, mrte_mel_lengths)

        if train_stage == "s1":
          logits, targets, loss_plm, loss_log_plm, acc_plm = net_plm(x_frame,lr_codes,w2v_lengths)
        else: #train_stage =="s1_1"
          logits, targets, loss_plm, loss_log_plm, acc_plm = net_plm1(x_frame,lr_codes,w2v_lengths)
        # print(f"x_frame, lr_codes: {x_frame.shape} {lr_codes.shape} {w2v_lengths} {x_frame[0:1,:,:].shape}")
        # codes = net_plm1.module.infer(x_frame[0:1,:,:])
          
    if train_stage == "s1":
      optim_p.zero_grad()
      scaler.scale(loss_log_plm).backward()
      scaler.unscale_(optim_p)
      grad_norm_p = commons.clip_grad_value_(net_plm.parameters(), None)
      scaler.step(optim_p)
      scaler.update()
    else:
      optim_p1.zero_grad()
      scaler.scale(loss_log_plm).backward()
      scaler.unscale_(optim_p1)
      grad_norm_p = commons.clip_grad_value_(net_plm1.parameters(), None)
      scaler.step(optim_p1)
      scaler.update()
    
    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        # losses = [loss_dur, loss_kl,loss_pitch,ctc_loss,loss_kl_r]
        losses = [loss_log_plm]
        logging.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logging.info([x.item() for x in losses] + [acc_plm, global_step, lr])
        
        scalar_dict = {"loss/g/log_plm": loss_log_plm, "learning_rate": lr, "grad_norm_p": grad_norm_p, "acc_plm": acc_plm}

        utils.summarize(
          writer=writer,
          global_step=global_step, 
          scalars=scalar_dict)

      # if global_step % hps.train.eval_interval == 0:
      #   evaluate(hps, net_g, eval_loader, writer_eval)
      if global_step % hps.train.save_interval == 0:
        if train_stage == "s1":
          utils.save_checkpoint(net_plm, optim_p, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "P_{}.pth".format(global_step)))
        else:
          utils.save_checkpoint(net_plm1, optim_p1, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "P1_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logging.info('====> Epoch: {}'.format(epoch))

                           
if __name__ == "__main__":
  main()
