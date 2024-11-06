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

from ttv_v1.t2w2v_transformer import SynthesizerTrn
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
  os.environ['MASTER_PORT'] = '53915'

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    # logger = utils.get_logger(hps.model_dir)
    import logging
    import coloredlogs
    coloredlogs.install(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            level=logging.INFO)
    logging.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)
  
  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)   #(text, mel, w2v, sid,phone_dur,pitch, rhy, language_id)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,150,250,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate()
  #text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid,dur_padded,pitch_padded, rhy_padded
  # collate_fn(train_dataset)
  train_loader = DataLoader(train_dataset, num_workers=4, shuffle=False, pin_memory=True,
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
  total_params = sum(p.numel() for p in net_g.parameters())
  print("Total parameters: ", total_params)
  
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
  net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
    global_step = int(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth").split('_')[-1].split('.')[0])
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logging, [writer, writer_eval], text_aligner)
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None, text_aligner)
    scheduler_g.step()
    # scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logging, writers, text_aligner):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step
  

  net_g.train()
  net_d.train()
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

    # print(f"x:{x.shape}, x_lengths:{x_lengths.shape}, mel_spk:{mel_spk.shape}, mel_spk_lengths:{mel_spk_lengths.shape} {mel_spk_lengths}, w2v:{w2v.shape}, w2v_lengths:{w2v_lengths.shape} {w2v_lengths}")
    # print(f"pitch:{pitch.shape}, pitch_lengths:{pitch_lengths.shape} {pitch_lengths}, tone:{tone.shape}, language:{language.shape}")
    # print(f"paths: {paths}")
    # print(f"dur: {dur.shape} {dur} {torch.sum(dur, dim=-1)}")

    ################## ASR ##################
    # with torch.no_grad():
    #   mask = length_to_mask(w2v_lengths).to('cuda')
    #   text_mask = length_to_mask(x_lengths).to(x.device)
    #   ppgs, s2s_pred, s2s_attn = text_aligner(w2v, mask, x)
    #   s2s_attn = s2s_attn.transpose(-1, -2)
    #   s2s_attn = s2s_attn[..., 1:]
    #   s2s_attn = s2s_attn.transpose(-1, -2)
    #   attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
    #   attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
    #   attn_mask = (attn_mask < 1)
    #   s2s_attn.masked_fill_(attn_mask, 0.0)
    #   mask_ST = mask_from_lens(s2s_attn, x_lengths, w2v_lengths)
    #   s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
    #   dur = torch.sum(s2s_attn_mono,dim=-1)

    with autocast(enabled=hps.train.fp16_run):
      l_length, l_pitch, x_mask, z_mask, pred_f0, w2v_pred, commit_loss, stats_ssl = net_g(x, x_lengths, w2v, w2v_lengths,mel_spk, mel_spk_lengths, pitch, pitch_lengths, tone, language, dur, mrte_mel, mrte_mel_lengths)  #生成器的输入增加了dur
      #print('ids_slice :',ids_slice.shape)
      # mel = spec_to_mel_torch(
      #     spec, 
      #     hps.data.filter_length, 
      #     hps.data.n_mel_channels, 
      #     hps.data.sampling_rate,
      #     hps.data.mel_fmin, 
      #     hps.data.mel_fmax)
      # # print('++++++mel_shape++++: ',mel.shape)
      # # print('++++++spec_shape++++: ',spec.shape)
      # y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      # # print("++++++=============y_hat: ",y_hat.shape)
      # y_hat_mel = mel_spectrogram_torch(
      #     y_hat.squeeze(1), 
      #     hps.data.filter_length, 
      #     hps.data.n_mel_channels, 
      #     hps.data.sampling_rate, 
      #     hps.data.hop_length, 
      #     hps.data.win_length, 
      #     hps.data.mel_fmin, 
      #     hps.data.mel_fmax
      # )

      # y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(w2v, w2v_pred.detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(w2v, w2v_pred)
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float()) * 2.0
        loss_pitch = torch.sum(l_pitch.float())   #添加loss_pitch
        # print('y_mel_shape: ',y_mel.shape)
        # print('y_hat_mel_shape: ',y_hat_mel.shape)
        # if y_mel.shape != y_hat_mel.shape:
        #   print('shape error:',y_mel.shape,y_hat_mel.shape)
        # loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        # loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
        l_w2v = (F.mse_loss(w2v, w2v_pred) * 1024 / torch.sum(z_mask)) * hps.train.c_mel
        l_w2v1 = (F.l1_loss(w2v, w2v_pred) * 1024 / torch.sum(z_mask)) * hps.train.c_mel
        # if z_r == None:
        #       loss_kl_r = 0
        # else:
        #       loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g) 
        commit_loss = commit_loss * hps.train.c_commit
        # loss_gen *= 0.25
        # ctc_loss = ctc_loss * hps.train.c_pho
        # loss_gen_all =  loss_dur + loss_kl + loss_pitch + ctc_loss + loss_kl_r
        loss_gen_all =  loss_dur + loss_pitch + l_w2v + l_w2v1 + loss_fm + loss_gen + commit_loss
    # print(loss_dur.dtype, loss_pitch.dtype, l_w2v.dtype)
    # print(f"loss_gen_all: {loss_gen_all.dtype}")
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    # print(f"net_g.parameters():\n {net_g.parameters()}")
    # for name,param in net_g.named_parameters():
    #   if param.grad is None:
    #     print(f"param name not update grad: {name}")

    # total = sum([param.nelement() for param in net_g.parameters()])
    # print("Number of parameter: %.2fM" % (total/1e6))

    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        # losses = [loss_dur, loss_kl,loss_pitch,ctc_loss,loss_kl_r]
        losses = [loss_dur, loss_pitch, l_w2v, l_w2v1, loss_fm, loss_gen, loss_disc, commit_loss]
        logging.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logging.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/dur": loss_dur, "loss/g/pitch": loss_pitch, "loss/g/l_w2v":l_w2v, "loss/g/l_w2v1":l_w2v1, "loss/g/l_commit": commit_loss})  #添加loss_pitch

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            # "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            # "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            # "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/f0": utils.plot_data_to_numpy(pitch[0, :].cpu().numpy(), pred_f0[0, :].detach().cpu().numpy()),  
            "all/stats_ssl": utils.plot_spectrogram_to_numpy(
                        stats_ssl[0].data.cpu().numpy()
                    ),
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
      if global_step % hps.train.save_interval == 0:
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logging.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    
    with torch.no_grad():
      for batch_idx, (x, x_lengths, mel_spk, mel_spk_lengths, w2v, w2v_lengths, pitch,pitch_lengths,tone,language, dur, paths, mrte_mel, mrte_mel_lengths) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        mel_spk, mel_spk_lengths = mel_spk.cuda(0), mel_spk_lengths.cuda(0)
        w2v, w2v_lengths = w2v.cuda(0), w2v_lengths.cuda(0)
        pitch, pitch_lengths = pitch.cuda(0), pitch_lengths.cuda(0)
        tone=tone.cuda(0)  #添加
        language=language.cuda(0)
        dur=dur.cuda(0)

        # remove else
        x = x[:1]
        tone = tone[:1]
        language = language[:1]
        x_lengths = x_lengths[:1]
        dur = dur[:1]
        # print(f"dur: {dur.shape} {x_lengths} {dur}")
        x = x[:,:x_lengths] # remove paded phone
        tone = tone[:,:x_lengths] # remove paded phone
        language = language[:,:x_lengths] # remove paded phone
        dur = dur[:,:x_lengths]
        mel_spk = mel_spk[:1]
        mel_spk_lengths = mel_spk_lengths[:1]
        mel_spk = mel_spk[:,:mel_spk_lengths]
        w2v = w2v[:1]
        w2v_lengths = w2v_lengths[:1]
        w2v = w2v[:,:w2v_lengths]
        pitch=pitch[:1]
        pitch_lengths=pitch_lengths[:1]
        pitch=pitch[:,:pitch_lengths]
        break
      w2v, pred_f0 = generator.module.infer(x, x_lengths, mel_spk, mel_spk_lengths, tone, language, dur=dur)
      
    image_dict = {
      f"all/f0-{batch_idx}": utils.plot_data_to_numpy(pitch[0, :].cpu().numpy(),
                                                       pred_f0[0, :].detach().cpu().numpy()),
     
    }

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      # audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  main()
