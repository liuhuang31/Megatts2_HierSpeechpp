import torch
from transformers import GPT2Config, GPT2Model

# 定义GPT-2的配置
config = GPT2Config(vocab_size=256, n_embd=768, n_layer=12, n_head=12)

# 创建GPT-2模型
model = GPT2Model(config)

# 输入数据
batch_size = 1
seq_length = 200
input_embeds = torch.randn(batch_size, seq_length, config.n_embd)

# 自回归训练
target = torch.randint(0, config.vocab_size, (batch_size, seq_length))
print(f"target: {target.shape} {target}")
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(inputs_embeds=input_embeds)
    logits = outputs.last_hidden_state
    print(f"logits1: {logits.shape}")
    logits = logits.reshape(-1, logits.size(-1))
    print(f"logits2: {logits.shape} {logits}")
    target = target.reshape(-1)
    print(f"target: {target.shape} {target}")
    loss = loss_fn(logits, target)
    loss.backward()
    optimizer.step()