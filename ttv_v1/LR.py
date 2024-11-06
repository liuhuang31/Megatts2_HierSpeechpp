class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.winlen = 1024
        self.hoplen = 256
        self.sr = 44100

    def LR(self, x, duration, x_lengths):
        output = list()
        frame_pitch = list()
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

            duration = predicted[i].item()
            if self.sr * duration - self.winlen > 0:
                expand_size = max((self.sr * duration - self.winlen)/self.hoplen, 1)
            elif duration == 0:
                expand_size = 0
            else:
                expand_size = 1
            vec_expand = vec.expand(max(int(expand_size), 0), -1)
            out.append(vec_expand)

        out = torch.cat(out, 0)
        return out

    def forward(self, x, duration, x_lengths):

        output, x_lengths = self.LR(x, duration, x_lengths)
        return output, x_lengths