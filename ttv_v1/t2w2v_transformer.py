import torch
from torch import nn
from torch.nn import functional as F
from ttv_v1 import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from commons import init_weights

import typing as tp
import transformers
import math
import random
from einops import rearrange
from ttv_v1.styleencoder import StyleEncoder
import commons
from ttv_v1.modules import WN
from ttv_v1.vits_models import LengthRegulator, DurationPredictor, LengthRegulator
from utils import length_to_mask, maximum_path
# from ttv_v1.utils_plm import topk_sampling, sample
# from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from ttv_v1.Gaussian import RangePredictor, GaussianUpsampling
from ttv_v1.quantize import ResidualVectorQuantizer
from torch.cuda.amp import autocast
from torch.nn import AvgPool1d, MaxPool1d
import numpy as np
from torchmetrics.classification import MulticlassAccuracy

#### GPT-SoVITS' AR modules
# from AR.models.utils import make_pad_mask
# from AR.models.utils import (
#     topk_sampling,
#     sample,
#     logits_to_probs,
#     multinomial_sample_one_no_sync,
# )
# from AR.modules.embedding import SinePositionalEmbedding
# from AR.modules.embedding import TokenEmbedding
# from AR.modules.transformer import LayerNorm
# from AR.modules.transformer import TransformerEncoder
# from AR.modules.transformer import TransformerEncoderLayer
# from torchmetrics.classification import MulticlassAccuracy

#### github megatts's modules
from ttv_v1.transformer_mega import TransformerEncoder, TransformerEncoderLayer
from ttv_v1.attentions import MultiHeadAttention

def get_2d_padding(kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)

class Wav2vec2(torch.nn.Module):
    def __init__(self, layer=7):

        """we use the intermediate features of xls-r-300m.
           More specifically, we used the output from the 12th layer of the 24-layer transformer encoder.
        """
        super().__init__()

        # self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/mms-300m")

        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()
        self.feature_layer = layer
    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (B x t)
        Returns:
            y: torch.Tensor of shape(B x C x t)
        """
        outputs = self.wav2vec2(x.squeeze(1), output_hidden_states=True)
        y = outputs.hidden_states[self.feature_layer]   
        y = y.permute((0, 2, 1))   
        return y

class TextEncoder(nn.Module):
  def __init__(self,
      n_vocab, n_tone, n_language,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.n_vocab = n_vocab
    self.n_tone = n_tone
    self.n_language = n_language
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    self.emb_tone = nn.Embedding(n_tone, hidden_channels)
    self.emb_language = nn.Embedding(n_language, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
    nn.init.normal_(self.emb_tone.weight, 0.0, hidden_channels**-0.5)
    nn.init.normal_(self.emb_language.weight, 0.0, hidden_channels**-0.5)
    self.cond = nn.Conv1d(256, hidden_channels, 1)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.encoder2 = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      1,
      kernel_size,
      p_dropout)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g, tone, language):
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    emb_tone = self.emb_tone(tone) * math.sqrt(self.hidden_channels) # [b, t, h]
    emb_language = self.emb_language(language) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = x + emb_tone + emb_language
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    
    x = self.encoder(x * x_mask, x_mask)
    
    # x = x + self.cond(g)
    x = self.encoder2(x * x_mask, x_mask)
    # stats = self.proj(x) * x_mask

    # m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, x_mask #m, logs, 

class MelEncoder(nn.Module):
  def __init__(self,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

  def forward(self, x, x_lengths):
    # x: # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.encoder(x * x_mask, x_mask)
    x = self.proj(x) * x_mask

    # m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, x_mask #m, logs, 
      

class W2VEncoder(nn.Module):
  def __init__(self,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.pro_hidden_channels = 1024

    self.cond = nn.Conv1d(256, hidden_channels, 1)
    self.project = nn.Conv1d(256, self.pro_hidden_channels, 1)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.encoder2 = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      1,
      kernel_size,
      p_dropout)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = x + self.cond(g)
    x = self.encoder(x * x_mask, x_mask)
    # x = self.project(x * x_mask)
    x = self.encoder2(x * x_mask, x_mask)
    return x, x_mask 
  

class ResidualCouplingBlock_Transformer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers=3,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels
    self.cond_block = torch.nn.Sequential(torch.nn.Linear(gin_channels, 4 * hidden_channels),
                                            nn.SiLU(), torch.nn.Linear(4 * hidden_channels, hidden_channels))

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer_Transformer_simple(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True, attention_head=4))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):

    g = self.cond_block(g.squeeze(2))

    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x
class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_mask, g=None):

    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs


class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1)
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw
  
class W2VDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 output_size=1024,
                 gin_channels=0,
                 p_dropout=0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, p_dropout=p_dropout)
        self.proj = nn.Conv1d(hidden_channels, output_size, 1)

    def forward(self, x, x_mask, g=None):
        x = self.pre(x * x_mask) * x_mask
        x = self.enc(x, x_mask, g=g)
        x = self.proj(x) * x_mask

        return x


class PitchPredictor(nn.Module):
  def __init__(self):
    super().__init__()

    resblock_kernel_sizes = [3,5,7]
    upsample_rates = [2,2]
    initial_channel = 1024
    upsample_initial_channel = 256
    upsample_kernel_sizes = [4,4]
    resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]

    self.num_kernels = len(resblock_kernel_sizes)
    self.num_upsamples = len(upsample_rates)
    self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

    resblock = modules.ResBlock1

    self.ups = nn.ModuleList()
    for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
        self.ups.append(weight_norm(
            ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                            k, u, padding=(k-u)//2)))

    self.resblocks = nn.ModuleList()
    for i in range(len(self.ups)):
        ch = upsample_initial_channel//(2**(i+1))
        for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
            self.resblocks.append(resblock(ch, k, d))

    self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
    self.ups.apply(init_weights)

    self.cond = Conv1d(256, upsample_initial_channel, 1)
    self.softplus = torch.nn.Softplus()

  def forward(self, x,  g):
    # 需要mask？ 上采样倍数增加
    x = self.conv_pre(x) + self.cond(g)

    for i in range(self.num_upsamples):
      x = F.leaky_relu(x, modules.LRELU_SLOPE)
      x = self.ups[i](x)
      xs = None
      for j in range(self.num_kernels):
        if xs is None:
          xs = self.resblocks[i*self.num_kernels+j](x)
        else:
          xs += self.resblocks[i*self.num_kernels+j](x)
      x = xs / self.num_kernels

    x = F.leaky_relu(x)
    ## Predictor
    x = self.conv_post(x)
    # x = self.softplus(x)

    return x


class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.x_scale = math.sqrt(dim_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x, offset = 0):
        """Reset the positional encodings."""
        x_size = x.size(1) + offset
        if self.pe is not None:
            if self.pe.size(1) >= x_size:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x_size, self.dim_model)
        if self.reverse:
            position = torch.arange(
                x_size - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(
                0, x_size, dtype=torch.float32
            ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor, offset : int = 0) -> torch.Tensor:
        self.extend_pe(x, offset)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, offset : x.size(1) + offset]
        return self.dropout(output)

  
class PLMConv(nn.Module):
  def __init__(self, hidden_channels=80):
    super().__init__()

    self.conv1 = Conv1d(hidden_channels, hidden_channels, 5, 1, padding=2)
    self.conv2 = Conv1d(hidden_channels, hidden_channels, 5, 1, padding=2)

  def forward(self, x, mel_mask):
    x = self.conv1(x * mel_mask) 
    x = self.conv2(x * mel_mask) 
    return x * mel_mask

## self.plm = attentions.Encoder
## n_layers=4,n_heads=4,vq_dim=20,tc_latent_dim=inter_channels,vq_bins=1024,kernel_size=9,dropout=0.2
class Megatts2PLM(nn.Module):
  def __init__(
    self,
    n_layers: int = 4,
    n_heads: int = 4,
    vq_dim: int = 20,
    tc_latent_dim: int = 256,
    vq_bins: int = 1024,
    kernel_size: int = 9,
    dropout: float = 0.1,
  ):
    super(Megatts2PLM, self).__init__()
    d_model = vq_dim + tc_latent_dim
    self.d_model = d_model
    # self.plm = TransformerEncoder(
    #     TransformerEncoderLayer(
    #         dim=d_model,
    #         ff_dim=d_model * 4,
    #         n_heads=n_heads,
    #         dropout=dropout,
    #         conv_ff=False,
    #     ),
    #     num_layers=n_layers,
    # )

    self.plm = attentions.Encoder(
      d_model,
      d_model * 4,
      n_heads,
      n_layers,
      kernel_size,
      dropout)
    
    self.predict_layer = nn.Linear(d_model, vq_bins, bias=False)

    self.pos_emb = SinePositionalEmbedding(d_model)
    self.pc_embedding = nn.Embedding(vq_bins + 2, vq_dim)

    self.ar_accuracy_metric = MulticlassAccuracy(
      vq_bins,
      top_k=10,
      average="micro",
      multidim_average="global",
      ignore_index=1024 + 1,
    )

  def pad_y_go(self, y, go_id):
    # go_id 设置和 vq_bins相关
    targets = F.pad(y, (1, 0), value=go_id)
    # 错位
    return targets[:, :-1], targets[:, 1:]
    
  def forward(
    self,
    tc_latent: torch.Tensor,  # (B, D, T)
    p_codes: torch.Tensor,  # (B, T)
    lens: torch.Tensor,  # (B,)
  ):
    # print(f"tc_latent, p_codes: {tc_latent.shape} {p_codes.shape}")
    tc_latent = tc_latent.transpose(-1, -2) # after transpose -> (B, T, D)
    
    p_codes, targets = self.pad_y_go(p_codes, go_id=1024) # targets [B, T]
    pc_emb = self.pc_embedding(p_codes)
    # print(f"tc_latent, pc_emb: {tc_latent.shape} {pc_emb.shape}")
    x_emb = torch.cat([tc_latent, pc_emb], dim=-1)
    x_pos = self.pos_emb(x_emb) # [b, t, h]
    x_pos = x_pos.transpose(-1,-2)  # [b, h, t]

    x_mask = torch.unsqueeze(commons.sequence_mask(lens, x_pos.size(2)), 1).to(tc_latent.dtype)
    x = self.plm(x_pos*x_mask, x_mask)  # input x's shape [b, h, t]
    logits = self.predict_layer(x.transpose(1,-1)).permute(0, 2, 1)
    # target = p_codes[:, 1:]
    loss = F.cross_entropy(logits, targets, reduction="sum", ignore_index=1024 + 1)
    loss_log = loss / targets.shape[0] / targets.shape[1]
    acc = self.ar_accuracy_metric(logits.detach(), targets).item()
    return logits, targets, loss, loss_log, acc

  def infer(
    self,
    tc_latent: torch.Tensor,  # (B, D, T)
  ):
    tc_latent = tc_latent.transpose(-1, -2) # after transpose -> (B, T, D)
    T = tc_latent.shape[1]
    p_code = torch.Tensor([1024]).to(
        tc_latent.device).type(torch.int64).unsqueeze(0)
    for t in range(T):
        pc_emb = self.pc_embedding(p_code)
        x_emb = torch.cat([tc_latent[:, 0:t+1, :], pc_emb], dim=-1)
        x_pos = self.pos_emb(x_emb)

        x = self.plm(x_pos)
        logits = self.predict_layer(x.transpose(1,-1)).permute(0, 2, 1)[:, -1:, :]
        p_code = torch.cat([p_code, logits.argmax(dim=-1)], dim=1)
    return p_code[:, 1:]

## self.plm = github_megatts2_transformer
class Megatts2PLM1(nn.Module):
  def __init__(
    self,
    n_layers: int = 4,
    n_heads: int = 4,
    vq_dim: int = 20,
    tc_latent_dim: int = 256,
    vq_bins: int = 1024,
    kernel_size: int = 9,
    dropout: float = 0.1,
  ):
    super(Megatts2PLM1, self).__init__()
    d_model = vq_dim + tc_latent_dim
    self.d_model = d_model
    self.plm = TransformerEncoder(
        TransformerEncoderLayer(
            dim=d_model,
            ff_dim=d_model * 4,
            n_heads=n_heads,
            dropout=dropout,
            conv_ff=False,
        ),
        num_layers=n_layers,
    )

    # self.plm = attentions.Encoder(
    #   d_model,
    #   d_model * 4,
    #   n_heads,
    #   n_layers,
    #   kernel_size,
    #   dropout)
    
    self.predict_layer = nn.Linear(d_model, vq_bins, bias=False)

    self.pos_emb = SinePositionalEmbedding(d_model)
    self.pc_embedding = nn.Embedding(vq_bins + 2, vq_dim)

    self.ar_accuracy_metric = MulticlassAccuracy(
      vq_bins,
      top_k=10,
      average="micro",
      multidim_average="global",
      ignore_index=1024 + 1,
    )

  def pad_y_go(self, y, go_id):
    # go_id 设置和 vq_bins相关
    targets = F.pad(y, (1, 0), value=go_id)
    # 错位
    return targets[:, :-1], targets[:, 1:]
    
  def forward(
    self,
    tc_latent: torch.Tensor,  # (B, D, T)
    p_codes: torch.Tensor,  # (B, T)
    lens: torch.Tensor,  # (B,)
  ):
    tc_latent = tc_latent.transpose(-1, -2) # after transpose -> (B, T, D)
    p_codes, targets = self.pad_y_go(p_codes, go_id=1024) # targets [B, T]
    pc_emb = self.pc_embedding(p_codes)

    x_emb = torch.cat([tc_latent, pc_emb], dim=-1)
    x_pos = self.pos_emb(x_emb) # [b, t, h]
    x = self.plm(x_pos, lens, causal=True)  # input x's shape [b, t, h]
    logits = self.predict_layer(x)
    # target = p_codes[:, 1:]

    logits = logits.transpose(1, 2)
    loss = F.cross_entropy(logits, targets, reduction="sum", ignore_index=1024 + 1)
    # loss_log = loss / targets.shape[0] / targets.shape[1]
    loss_log = loss / torch.sum(lens)
    acc = self.ar_accuracy_metric(logits.detach(), targets).item()
    return logits, targets, loss, loss_log, acc
  
  def infer(
    self,
    tc_latent: torch.Tensor,  # (B, D, T)
  ):
    tc_latent = tc_latent.transpose(-1, -2) # after transpose -> (B, T, D)
    T = tc_latent.shape[1]
    p_code = torch.Tensor([1024]).to(
        tc_latent.device).type(torch.int64).unsqueeze(0)
    for t in range(T):
        pc_emb = self.pc_embedding(p_code)
        x_emb = torch.cat([tc_latent[:, 0:t+1, :], pc_emb], dim=-1)
        x_pos = self.pos_emb(x_emb)

        x = self.plm(x_pos)
        logits = self.predict_layer(x)[:, -1:, :]
        p_code = torch.cat([p_code, logits.argmax(dim=-1)], dim=1)
    return p_code[:, 1:]
  

class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self,
    n_vocab, n_tone, n_language,
    # text_aligner,
    spec_channels,
    hop_length,
    sampling_rate,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock,
    resblock_kernel_sizes,
    resblock_dilation_sizes,
    gin_channels=256,
    prosody_size=20,
    cfg=False,
    freeze_quantizer=None,
    **kwargs):

    super().__init__()
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.segment_size = segment_size
    self.mel_size = prosody_size
    self.stride = 8
    # self.text_aligner = text_aligner

    # self.enc_q = PosteriorEncoder(1024, inter_channels, hidden_channels, 5, 1, 16,  gin_channels=256)
    self.enc_p = TextEncoder(n_vocab, n_tone, n_language, out_channels=inter_channels, hidden_channels=inter_channels, filter_channels=inter_channels*4, 
                                 n_heads=4, n_layers=3, kernel_size=9, p_dropout=0.2)
    
    #### mrte
    self.mel_encoder = MelEncoder(out_channels=256, hidden_channels=80, filter_channels=80*4, 
                                 n_heads=4, n_layers=2, kernel_size=9, p_dropout=0.2)
    self.mha = MultiHeadAttention(inter_channels, inter_channels, n_heads=4, p_dropout=0.2)
    self.cond_g = nn.Conv1d(256, inter_channels, 1)
    # self.flow = ResidualCouplingBlock_Transformer(inter_channels, hidden_channels, 5, 1, 3, gin_channels=256)

    self.w2v_encoder = W2VEncoder(out_channels=inter_channels, hidden_channels=inter_channels, filter_channels=inter_channels*4, 
                                 n_heads=4, n_layers=3, kernel_size=9, p_dropout=0.2)
    self.w2v_decoder = W2VDecoder(inter_channels, inter_channels*2, 5, 1, 8, output_size=1024, p_dropout=0.1, gin_channels=256)    
    
    self.emb_g = StyleEncoder(in_dim=80, hidden_dim=256, out_dim=256)
    # self.dp = StochasticDurationPredictor(inter_channels, inter_channels, 3, 0.5, 4, gin_channels=256)

    self.duration_predictor = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)
    self.RangePredictor = RangePredictor(257,256)
    self.gaussian = GaussianUpsampling()
    self.lr = LengthRegulator(hop_length, sampling_rate)  #####添加LR
    self.dur_downsample = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, stride=2, padding=0)

    self.pp = PitchPredictor()
    # self.phoneme_classifier = Conv1d(inter_channels, 178, 1, bias=False)

    # use mel VQ, we use 20bins
    self.plm_conv1 = PLMConv(hidden_channels=20)
    self.vq_pooling = MaxPool1d(kernel_size=8, stride=8)
    self.plm_conv2 = PLMConv(hidden_channels=20)
    self.quantizer = ResidualVectorQuantizer(dimension=20, n_q=1, bins=1024)
    if freeze_quantizer:
      print(f"-------------- freeze_quantizer --------------")
      # self.ssl_proj.requires_grad_(False) # this for semantic_frame_rate, not use here
      self.quantizer.requires_grad_(False)
    self.ssl_proj = nn.Conv1d(20, inter_channels, 1)

    # megatts2 plm
    # self.plm = Megatts2PLM(n_layers=4,n_heads=4,vq_dim=20,tc_latent_dim=inter_channels,vq_bins=1024,kernel_size=9,dropout=0.2)

  def forward(self, x, x_lengths, w2v, w2v_lengths, mel_spk, mel_spk_lengths,  pitch, pitch_lengths, tone, language,dur,mrte_mel, mrte_mel_lengths):
    mel_mask = torch.unsqueeze(commons.sequence_mask(mel_spk_lengths, mel_spk.size(2)), 1).to(mel_spk.dtype)
    mrte_mel_mask = torch.unsqueeze(commons.sequence_mask(mrte_mel_lengths, mrte_mel.size(2)), 1).to(mrte_mel.dtype)
    mel_pool_mask = torch.unsqueeze(commons.sequence_mask(mel_spk_lengths/8, mel_spk.size(2)/8), 1).to(mel_spk.dtype)
    pitch_mask = torch.unsqueeze(commons.sequence_mask(pitch_lengths, pitch.size(1)), 1).to(pitch.dtype)
    mel_len = mel_spk.size(-1)
    
    ################## emb_g and text_encoder ##################
    g = self.emb_g(mrte_mel, mrte_mel_mask).unsqueeze(-1)
    x, x_mask = self.enc_p(x, x_lengths, g, tone, language)
    x_mask_ori = x_mask

    ################## mrte ##################
    mel_encoder_output, h_mask = self.mel_encoder(mrte_mel, mrte_mel_lengths)
    # h_mask: [2, 1, 184]; x_mask: [2, 1, 19];
    # mha_attn_mask = [2, 1, 1, 184] * [2, 1, 19, 1] = [2, 1, 19, 184]
    mha_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    mha_output = self.mha(x, mel_encoder_output, mha_attn_mask)
    cond_g = self.cond_g(g)
    x = x + mha_output + cond_g

    ## 2. for maybe use gaussian upsample, use the LR way.
    ## But if not use gaussion upsample, its same as way 1.
    logw_ = torch.log(dur.detach().float() + 1).unsqueeze(1) * x_mask
    logw = self.duration_predictor(x, x_mask, g=g)  # 
    l_loss = torch.sum((logw - logw_) ** 2, [1, 2])
    x_mask_sum = torch.sum(x_mask)
    l_length = l_loss / x_mask_sum

    ## gaussian 
    range = self.RangePredictor(x,dur,x_lengths)
    range = torch.minimum(range, dur*2)
    range = torch.maximum(range, torch.tensor(1e-5))
    x_frame = self.gaussian(x, dur, range, x_lengths).to(x.dtype)
    x_frame = self.dur_downsample(x_frame).to(x.dtype)

    ################## VQ ##################
    with autocast(enabled=False):
      mel_spk = mel_spk[:,:20,:]
      mel_spk = self.plm_conv1(mel_spk, mel_mask)
      mel_spk = self.vq_pooling(mel_spk)
      mel_spk = self.plm_conv2(mel_spk, mel_pool_mask)
      quantized, codes, commit_loss, quantized_list = self.quantizer(
        mel_spk, layers=[0]
      )
    quantized = rearrange(quantized, "B D T -> B T D").unsqueeze(2).contiguous().expand(-1, -1, self.stride, -1)
    quantized = rearrange(quantized, "B T S D -> B (T S) D")[:, :mel_len, :] 
    quantized = quantized.transpose(1,-1) * mel_mask

    ################## plm ##################
    quantized_proj = self.ssl_proj(quantized)
    quantized_proj = quantized_proj * mel_mask
    x_frame = x_frame + quantized_proj

    ################## w2v_encoder and pitch_predictor ##################
    x2v_enc, y_mask = self.w2v_encoder(x_frame, w2v_lengths, g=g)
    w2v_pred = self.w2v_decoder(x2v_enc, y_mask, g=g) # w2v_pred: torch.Size([1, 1024, 200])

    # relu activation maybe cause some problem
    # whether use truth or predict w2v
    f0_prob = random.random()
    if f0_prob > 0.5:
      pred_lf0 = self.pp(w2v_pred, g).squeeze(1) * pitch_mask.squeeze(1)
    else:
      pred_lf0 = self.pp(w2v, g).squeeze(1) * pitch_mask.squeeze(1)
    LF0 = torch.log(pitch+1)
    l_pitch = F.l1_loss(pred_lf0, LF0)

    pred_f0 = torch.exp(pred_lf0)  #pred_lf0 * 800.

    # print(f"w2v_pred: {w2v_pred.dtype}, pred_f0: {pred_f0.dtype}")
    # print(f"pred_f0: {pred_f0.shape} w2v_pred: {w2v_pred.shape}")
    # save_w2v = w2v_pred[0][:,:mel_spk_lengths[0]].detach().cpu().numpy()
    # save_f0 = pitch[0][:mel_spk_lengths[0]*4].cpu().detach().numpy()
    # print(f"save_w2v: {mel_spk_lengths[0]} {save_w2v.shape}, save_f0: {save_f0.shape}")
    # np.save("/home/liuhuang/workspace/llm/HierSpeechpp_exp15_plm/tt.w2v.npy", save_w2v)
    # np.save("/home/liuhuang/workspace/llm/HierSpeechpp_exp15_plm/tt.f0.npy", save_f0)

    return l_length, l_pitch, x_mask_ori, y_mask, pred_f0, w2v_pred, commit_loss, quantized
  
  @torch.no_grad()
  def extract_tc_latent_code(self, x, x_lengths, mel_spk, mel_spk_lengths, tone, language,dur, mrte_mel, mrte_mel_lengths):
    mel_mask = torch.unsqueeze(commons.sequence_mask(mel_spk_lengths, mel_spk.size(2)), 1).to(mel_spk.dtype)
    mrte_mel_mask = torch.unsqueeze(commons.sequence_mask(mrte_mel_lengths, mrte_mel.size(2)), 1).to(mrte_mel.dtype)
    mel_pool_mask = torch.unsqueeze(commons.sequence_mask(mel_spk_lengths/8, mel_spk.size(2)/8), 1).to(mel_spk.dtype)
    ################## emb_g and text_encoder ##################
    g = self.emb_g(mrte_mel, mrte_mel_mask).unsqueeze(-1)
    x, x_mask = self.enc_p(x, x_lengths, g, tone, language)

    ################## mrte ##################
    mel_encoder_output, h_mask = self.mel_encoder(mrte_mel, mrte_mel_lengths)
    # h_mask: [2, 1, 184]; x_mask: [2, 1, 19];
    # mha_attn_mask = [2, 1, 1, 184] * [2, 1, 19, 1] = [2, 1, 19, 184]
    mha_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    mha_output = self.mha(x, mel_encoder_output, mha_attn_mask)
    cond_g = self.cond_g(g)
    x = x + mha_output + cond_g

    ## gaussian 
    range = self.RangePredictor(x,dur,x_lengths)
    range = torch.minimum(range, dur*2)
    range = torch.maximum(range, torch.tensor(1e-5))
    x_frame = self.gaussian(x, dur, range, x_lengths).to(x.dtype)
    x_frame = self.dur_downsample(x_frame).to(x.dtype)

    ################## VQ ##################
    with autocast(enabled=False):
      mel_spk = mel_spk[:,:20,:]
      mel_spk = self.plm_conv1(mel_spk, mel_mask)
      mel_spk = self.vq_pooling(mel_spk)
      mel_spk = self.plm_conv2(mel_spk, mel_pool_mask)
      quantized, codes, commit_loss, quantized_list = self.quantizer(
        mel_spk, layers=[0]
      )
      quantized = quantized * mel_pool_mask
    
    ################## plm ##################
    # ## lr_codes for trin plm, input_codes [1, 8, 18]
    codes = codes.squeeze(0).unsqueeze(1) # [8, 1, 18]
    codes = rearrange(codes, "B D T -> B T D").unsqueeze(2).contiguous().expand(-1, -1, self.stride, -1) #[8, 18, 8, 1]
    codes = rearrange(codes, "B T S D -> B (T S) D")  # [8, 144, 1]
    codes = codes.transpose(1,-1) * mel_mask
    lr_codes = codes.squeeze(1).long()  # [8, 1, 144]
    return x_frame, lr_codes
  
  def extract_latent(self, x):
    # x = self.ssl_proj(x)
    quantized, codes, commit_loss, quantized_list = self.quantizer(x)
    return codes.transpose(0, 1)
  
  @torch.no_grad()
  def inf_extract_tc_latent(self, x, x_lengths, y_mel, y_length, tone, language, mrte_mel=None, mrte_mel_lengths=None, length_scale=1):
    mel_mask = torch.unsqueeze(commons.sequence_mask(y_length, y_mel.size(2)), 1).to(y_mel.dtype)

    ################## emb_g and text_encoder ##################
    if mrte_mel is not None:
      mrte_mel_mask = torch.unsqueeze(commons.sequence_mask(mrte_mel_lengths, mrte_mel.size(2)), 1).to(mrte_mel.dtype)
      g = self.emb_g(mrte_mel, mrte_mel_mask).unsqueeze(-1)
    else:
      g = self.emb_g(y_mel, mel_mask).unsqueeze(-1)
    x, x_mask = self.enc_p(x, x_lengths, g, tone, language)

    ################## mrte ##################
    if mrte_mel is not None:
      mel_encoder_output, h_mask = self.mel_encoder(mrte_mel, mrte_mel_lengths)
    else:
      mel_encoder_output, h_mask = self.mel_encoder(y_mel, y_length)
    # h_mask: [2, 1, 184]; x_mask: [2, 1, 19];
    # mha_attn_mask = [2, 1, 1, 184] * [2, 1, 19, 1] = [2, 1, 19, 184]
    mha_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    mha_output = self.mha(x, mel_encoder_output, mha_attn_mask)
    cond_g = self.cond_g(g)
    x = x + mha_output + cond_g

    logw = self.duration_predictor(x, x_mask, g=g)
    w = torch.exp(logw) * x_mask * length_scale
    dur = torch.ceil(w)
    
    # print(f"x,dur: {x.shape} {dur.shape}")
    ## gaussian 
    range = self.RangePredictor(x,dur.squeeze(1),x_lengths)
    range = torch.minimum(range, dur*2)
    range = torch.maximum(range, torch.tensor(1e-5))
    x_frame = self.gaussian(x, dur.squeeze(1), range, x_lengths).to(x.dtype)
    x_frame = self.dur_downsample(x_frame).to(x.dtype)

    x_frame = x_frame.to(x.device)

    frame_lengths = list()
    for d, x_l in zip(dur, x_lengths):
      frame_lengths.append(torch.sum(d[:x_l], dtype=torch.long))
    x_lengths = torch.LongTensor(frame_lengths).to(x.device)
    x_lengths = x_lengths / 2
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x_frame.size(2)), 1).contiguous().to(x.device)

    return x_frame, g, x_lengths, x_mask
  
  @torch.no_grad()
  def inf_plm_gen(self, x_frame, g, codes, x_lengths, x_mask):
    quantized = self.quantizer.decode(codes)
    quantized_proj = self.ssl_proj(quantized)
    x_frame = x_frame + quantized_proj

    x2v_enc, y_mask = self.w2v_encoder(x_frame, x_lengths, g=g)
    w2v_pred = self.w2v_decoder(x2v_enc, x_mask, g=g) # w2v_pred: torch.Size([1, 1024, 200])
    pred_lf0 = self.pp(w2v_pred, g).squeeze(1)
    # pred_f0 = torch.exp(pred_lf0)
    return w2v_pred, pred_lf0
  
  @torch.no_grad()
  def infer(self, x, x_lengths, mel_spk, mel_spk_lengths, tone, language, dur=None, mrte_mel=None, mrte_mel_lengths=None, noise_scale=1, noise_scale_w=1, length_scale=1, denoise_ratio = 0):
    mel_mask = torch.unsqueeze(commons.sequence_mask(mel_spk_lengths, mel_spk.size(2)), 1).to(mel_spk.dtype)
    mel_pool_mask = torch.unsqueeze(commons.sequence_mask(mel_spk_lengths/8, mel_spk.size(2)/8), 1).to(mel_spk.dtype)
    mel_len = mel_spk.size(-1)

    ################## emb_g and text_encoder ##################
    if mrte_mel is not None:
      mrte_mel_mask = torch.unsqueeze(commons.sequence_mask(mrte_mel_lengths, mrte_mel.size(2)), 1).to(mrte_mel.dtype)
      g = self.emb_g(mrte_mel, mrte_mel_mask).unsqueeze(-1)
    else:
      g = self.emb_g(mel_spk, mel_mask).unsqueeze(-1)
    g = self.emb_g(mel_spk, mel_mask).unsqueeze(-1)
    x, x_mask = self.enc_p(x, x_lengths, g, tone, language)

    ################## mrte ##################
    if mrte_mel is not None:
      mel_encoder_output, h_mask = self.mel_encoder(mrte_mel, mrte_mel_lengths)
    else:
      mel_encoder_output, h_mask = self.mel_encoder(mel_spk, mel_spk_lengths)
    # h_mask: [2, 1, 184]; x_mask: [2, 1, 19];
    # mha_attn_mask = [2, 1, 1, 184] * [2, 1, 19, 1] = [2, 1, 19, 184]
    mha_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    mha_output = self.mha(x, mel_encoder_output, mha_attn_mask)
    cond_g = self.cond_g(g)
    x = x + mha_output + cond_g

    logw = self.duration_predictor(x, x_mask, g=g)  # 
    w = torch.exp(logw) * x_mask
    duration = torch.ceil(w)
    if dur is None:
      dur = duration

    ## gaussian 
    range = self.RangePredictor(x,dur,x_lengths)
    range = torch.minimum(range, dur*2)
    range = torch.maximum(range, torch.tensor(1e-5))
    x_frame = self.gaussian(x, dur, range, x_lengths).to(x.dtype)
    x_frame = self.dur_downsample(x_frame).to(x.dtype)

    frame_lengths = list()
    for d, x_l in zip(duration, x_lengths):
      frame_lengths.append(torch.sum(d[:x_l], dtype=torch.long))
    x_lengths = torch.LongTensor(frame_lengths).to(x.device)
    x_lengths = x_lengths / 2

    ################## VQ ##################
    with autocast(enabled=False):
      mel_spk = mel_spk[:,:20,:]
      mel_spk = self.plm_conv1(mel_spk, mel_mask)
      mel_spk = self.vq_pooling(mel_spk)
      mel_spk = self.plm_conv2(mel_spk, mel_pool_mask)
      quantized, codes, commit_loss, quantized_list = self.quantizer(
        mel_spk, layers=[0]
      )
    quantized = rearrange(quantized, "B D T -> B T D").unsqueeze(2).contiguous().expand(-1, -1, self.stride, -1)
    quantized = rearrange(quantized, "B T S D -> B (T S) D")[:, :mel_len, :] 
    quantized = quantized.transpose(1,-1) * mel_mask

    quantized_proj = self.ssl_proj(quantized)
    quantized_proj = quantized_proj * mel_mask
    x_frame = x_frame + quantized_proj

    ################## w2v_encoder and pitch_predictor ##################
    x2v_enc, y_mask = self.w2v_encoder(x_frame, x_lengths, g=g)
    # print(f"x2v_enc: {x2v_enc.dtype}")
    # print(f"x2v_enc: {x2v_enc.shape}")  # x2v_enc: torch.Size([1, 256, 200])
    # print(f"x2v_enc, y_mask: {x2v_enc.shape} {y_mask.shape}")
    w2v_pred = self.w2v_decoder(x2v_enc, y_mask, g=g) # w2v_pred: torch.Size([1, 1024, 200])
    pred_lf0 = self.pp(w2v_pred, g).squeeze(1)
    # pred_f0 = torch.exp(pred_lf0)  #pred_lf0 * 800.

    # print(f"w2v_pred inf: {w2v_pred.dtype}, pred_f0: {pred_f0.dtype}")
    # print(f"pred_f0 inf: {pred_f0.shape} w2v_pred: {w2v_pred.shape}")
    # save_w2v = w2v_pred[0][:,:mel_spk_lengths[0]].detach().cpu().numpy()
    # save_f0 = pred_f0[0][:mel_spk_lengths[0]*4].cpu().detach().numpy()
    # print(f"save_w2v inf: {mel_spk_lengths[0]} {save_w2v.shape}, save_f0: {save_f0.shape}")
    
    # np.save("/home/liuhuang/workspace/llm/HierSpeechpp_exp15_plm/tt_inf.w2v.npy", save_w2v)
    # np.save("/home/liuhuang/workspace/llm/HierSpeechpp_exp15_plm/tt_inf.f0.npy", save_f0)

    return w2v_pred, pred_lf0
  
  @torch.no_grad()
  def infer_noise_control(self, x, x_lengths, y_mel, y_length, noise_scale=0.333, noise_scale_w=1, length_scale=1, denoise_ratio = 0):

    y_mask = torch.unsqueeze(commons.sequence_mask(y_length, y_mel.size(2)), 1).to(y_mel.dtype)

    # Speaker embedding from mel (Style Encoder)
    g = self.emb_g(y_mel, y_mask).unsqueeze(-1)
    
    g_org, g_denoise = g[:1, :, :], g[1:, :, :]
    g = (1-denoise_ratio)*g_org + denoise_ratio*g_denoise


    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)


    logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
      
    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, reverse=True)

    w2v = self.w2v_decoder(z, y_mask, g=g)
    pitch = self.pp(w2v, g)

    return w2v, pitch
