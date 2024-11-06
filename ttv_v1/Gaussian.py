import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

class LinearNorm(nn.Module):
    """
        linear layer with xavier initialization
        https://github.com/NVIDIA/tacotron2/blob/185cd24e046cc1304b4f8e564734d2498c6e2e6f/layers.py#L8
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)
    
class GaussianUpsampling(nn.Module):
    """
        Non-attention Tacotron:
            - https://arxiv.org/abs/2010.04301
        this source code is implemenation of the ExpressiveTacotron from BridgetteSong
            - https://github.com/BridgetteSong/ExpressiveTacotron/blob/master/model_duration.py
    """
    def __init__(self):
        super(GaussianUpsampling, self).__init__()
        self.mask_score = -1e15

    def forward(self, encoder_outputs, durations, vars, input_lengths=None):
        """ Gaussian upsampling
        PARAMS
        ------
        encoder_outputs: Encoder outputs  [B, N, H]
        durations: phoneme durations  [B, N]
        vars : phoneme attended ranges [B, N]
        input_lengths : [B]
        RETURNS
        -------
        encoder_upsampling_outputs: upsampled encoder_output  [B, T, H]
        """
        encoder_outputs=encoder_outputs.transpose(1,2)
    
        B = encoder_outputs.size(0)
        N = encoder_outputs.size(1)
        m=torch.sum(durations, dim=1)
        T = int(torch.sum(durations, dim=1).max().item())
        c = torch.cumsum(durations, dim=1).float() - 0.5 * durations
        c = c.unsqueeze(2) # [B, N, 1]
        t = torch.arange(T, device=encoder_outputs.device).expand(B, N, T).float()  # [B, N, T]
        vars = vars.view(B, -1, 1) # [B, N, 1]


        w_t = -0.5 * (np.log(2.0 * np.pi) + torch.log(vars) + torch.pow(t - c, 2) / vars) # [B, N, T]

        if input_lengths is not None:
            input_masks = ~self.get_mask_from_lengths(input_lengths, N) # [B, N]
            input_masks = torch.tensor(input_masks, dtype=torch.bool, device=w_t.device)
            masks = input_masks.unsqueeze(2)
            w_t.data.masked_fill_(masks, self.mask_score)
        w_t = F.softmax(w_t, dim=1)
        encoder_upsampling_outputs = torch.bmm(w_t.transpose(1, 2), encoder_outputs)  # [B, T, encoder_hidden_size]

        return encoder_upsampling_outputs.transpose(1,2)

    
    def get_mask_from_lengths(self, lengths, max_len=None):
        if max_len is None:
            max_len = max(lengths)
        ids = torch.tensor(np.arange(0, max_len),device=lengths.device)
        mask = (ids < lengths.reshape(-1, 1))
        return mask
    
class RangePredictor(nn.Module):
    """Duration Predictor module:
        - two stack of BiLSTM
        - one projection layer
    """
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.lstm = nn.LSTM(in_channel,out_channel, 1, batch_first=True, bidirectional=True)
        self.proj = LinearNorm(out_channel*2,1)
        

    def forward(self, encoder_outputs, durations, input_lengths=None):
        """
            :param encoder_outputs:
            :param durations:
            :param input_lengths:
            :return:
        """
        # print(f"encoder_outputs, durations: {encoder_outputs.shape} {durations.shape} {input_lengths.shape}")
        encoder_outputs = encoder_outputs.transpose(1, 2)
        concated_inputs = torch.cat([encoder_outputs, durations.unsqueeze(-1)], dim=-1)

        ## remove pad activations 
        if input_lengths is not None:
            concated_inputs = pack_padded_sequence(
                concated_inputs, input_lengths.cpu(), batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(concated_inputs)

        if input_lengths is not None:
            outputs, _ = pad_packed_sequence(
                outputs, batch_first=True)

        outputs = self.proj(outputs)
        outputs = F.softplus(outputs)
        return outputs.squeeze()
