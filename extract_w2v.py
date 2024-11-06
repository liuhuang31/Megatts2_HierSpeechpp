import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torchaudio
import librosa
import torch
from tqdm import tqdm
import argparse
from torch.nn import functional as F
import transformers

# print(source_audio.shape, source_audio[0:20])
# b, sr = librosa.load(wavpath, sr=16000)
# print(b.shape, b[0:20])


class Wav2vec2(torch.nn.Module):
    def __init__(self, layer=7, w2v='mms'):

        """we use the intermediate features of mms-300m.
           More specifically, we used the output from the 7th layer of the 24-layer transformer encoder.
        """
        super().__init__()

        if w2v == 'mms':
           self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("/home/liuhuang/workspace/llm/facebook/mms-300m")
        else:
           self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")

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
        y = outputs.hidden_states[self.feature_layer]  # B x t x C(1024)
        y = y.permute((0, 2, 1))  # B x t x C -> B x C x t
        return y


def find_all_wav_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            if not os.path.splitext(filename)[-1] == '.wav':
                # print('the file:{} is not a wav file,skip it!'.format(filename))
                continue
            apath = os.path.join(maindir, filename) # merge into a full path
            result.append(apath)
    return result


def __cmd():
    description = "extract w2v"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--input_dir",
        type=str,
        default='',
        required=False,
        help="the audio corpus dir.")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        raise ValueError(f"input_dir not exists: {input_dir}")
    w2v = Wav2vec2().cuda()
    wav_lists = find_all_wav_path(input_dir)

    for item in tqdm(wav_lists):
        w2v_path = item.replace('.wav','.hw2v.pt')
        if os.path.exists(w2v_path):
            continue
        try:
            source_audio, sample_rate = torchaudio.load(item)
            if sample_rate != 16000:
                source_audio = torchaudio.functional.resample(source_audio, sample_rate, 16000, resampling_method="kaiser_window")  # sinc_interpolation  kaiser_window
        except Exception as e:
            print(f"{item} {e}")
            continue
        p = (source_audio.shape[-1] // 1280 + 1) * 1280 - source_audio.shape[-1]
        source_audio = torch.nn.functional.pad(source_audio, (0, p), mode='constant').data
        y_pad = F.pad(source_audio, (40, 40), "reflect")

        # print(f"{w2v_path} {item}")
        # exit()
        x_w2v = w2v(y_pad.cuda()).cpu()
        torch.save(x_w2v, w2v_path)
    
  
if __name__ == '__main__':
    __cmd()

'''
# zhvoice zhthchs30
CUDA_VISIBLE_DEVICES="0" python extract_w2v.py --input_dir /data2/liuhuang/VCTK-Corpus-Processed
'''