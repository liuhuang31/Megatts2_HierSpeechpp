import torch
from torch import nn
from torch.nn import functional as F
import ttv_v1.attentions as attentions
from ttv_v1 import modules

class FramePriorNet(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        # self.emb = nn.Embedding(121, hidden_channels)  #这行没有用到

        self.fft_block = attentions.FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

    def forward(self, x_frame, x_mask):
        x = x_frame
        #print('-------------x.shape; x_mask.shape,(x * x_mask.shape)',x.shape,x_mask.shape,(x * x_mask).shape)
        x = self.fft_block(x * x_mask, x_mask)
        x = x.transpose(1, 2)
        return x

class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self, hoplen, sr):
        super(LengthRegulator, self).__init__()
        self.hoplen = hoplen
        self.sr = sr

    def LR(self, x, duration, x_lengths):
        output = list()
        mel_len = list()
        x = torch.transpose(x, 1, -1)
        frame_lengths = list()

        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            frame_lengths.append(expanded.shape[0])

        max_len = max(frame_lengths)
        output_padded = torch.FloatTensor(x.size(0), max_len, x.size(2))
        output_padded.zero_()
        for i in range(output_padded.size(0)):
            output_padded[i, :frame_lengths[i], :] = output[i]
        output_padded = torch.transpose(output_padded, 1, -1)

        return output_padded, torch.LongTensor(frame_lengths)

    def expand(self, batch, predicted):
        out = list()
        predicted = predicted.squeeze()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            vec_expand = vec.expand(max(int(expand_size), 0), -1)
            out.append(vec_expand)

        out = torch.cat(out, 0)
        return out

    def forward(self, x, duration, x_lengths):

        output, x_lengths = self.LR(x, duration, x_lengths)
        return output, x_lengths
    
class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    
    self.lstms=nn.LSTM(in_channels,filter_channels,num_layers=2,bidirectional=True,batch_first=True)
    self.norm_2 = modules.LayerNorm(filter_channels*2)
    
    
    self.proj = nn.Conv1d(filter_channels*2, 1, 1)
    # self.proj = nn.Linear(filter_channels*2, 1)
    self.softplus = torch.nn.Softplus()

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None):
    # x = torch.detach(x)
    if g is not None:
    #   g = torch.detach(g)
      # print('++++++x_shape+++++:',x.shape)   #[32, 192, 129]
      # print('+++++++g_shape++++:',g.shape)   #[32, 256, 1])
      # print('++++++cond(g)++++:',self.cond(g).shape)
      x = x + self.cond(g)  

    x = torch.transpose(x, 1, -1) # [b, h, t] -> [b, t, h]
    x_mask=torch.transpose(x_mask, 1, -1)
    x , _= self.lstms(x*x_mask)    #x.shape[1,129,512]

    ## for lt conv project
    x=torch.transpose(x, 1, -1)
    x_mask=torch.transpose(x_mask, 1, -1)
    x = self.norm_2(x)  #先norm再relu
    x = torch.relu(x)
    x = self.drop(x)  #[1,512,129]
    x = self.proj(x * x_mask)  #[1,1,129]
    x = self.softplus(x)
    x = x * x_mask
    
    ### for liner project
    # x = self.drop(x)  #[1,512,129]
    # x = self.proj(x * x_mask)  #[1,1,129]
    # x = self.softplus(x)
    # x = x * x_mask
    # x = torch.transpose(x, 1, -1)

    return x

def max_norm(x,y):
   #x是dur, y是range, return min(y, log(2*x))
   x=x.squeeze().squeeze()
   x=x.float()
   mask=torch.gt(y,2*x)
   c=torch.where(mask,2*x,y)
   return c
def min_norm(x):
   #return max(x,1e-5)
   mask=torch.lt(x,1e-5)
   a = torch.where(mask, torch.Tensor([1e-5]).to(device=x.device, dtype=x.dtype), x)
   return a
    
class Projection(nn.Module):
    def __init__(self,
                 hidden_channels,
                 out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask):
        stats = self.proj(x) * x_mask  #stats.shape[1,192*2,1598（帧长）]
        m_p, logs_p = torch.split(stats, self.out_channels, dim=1)
        return m_p, logs_p
