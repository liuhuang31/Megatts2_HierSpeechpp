import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d, AvgPool2d
import torch.nn.functional as F
LRELU_SLOPE = 0.1

class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self,use_spectral_norm=False):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        
        self.discriminators = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            # norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))),
        ])

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):

        fmap = []
        # y = y.squeeze(1)
        # y = stft(y, self.fft_size, self.shift_size, self.win_length, self.window.to(y.get_device()))
        
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap

class MultiResSpecDiscriminator(torch.nn.Module):

    def __init__(self):

        super(MultiResSpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            SpecDiscriminator(use_spectral_norm=True),
            SpecDiscriminator(),
            # SpecDiscriminator()
            ])
        self.meanpools = nn.ModuleList([
            AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        y = y.unsqueeze(1)
        y_hat = y_hat.unsqueeze(1)
        for i, d in enumerate(self.discriminators):
            if i !=0:
              y = self.meanpools[i-1](y)
              y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs