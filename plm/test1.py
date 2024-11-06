import torch
text_emb = torch.randn(1, 10, 1024)
text_start = text_end = torch.randn(1, 1, 1024)
text_emb = torch.cat((text_start, text_emb, text_end), dim=1)
print(text_emb.shape)

text_lengths = torch.LongTensor([text_emb.shape[1]])
audio_codes = torch.randint(low=0, high=10, size=(10,), dtype=torch.int32)
print(audio_codes)