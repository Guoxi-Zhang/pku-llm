

from dataclasses import dataclass
import torch
from torch import  nn
from torch.nn import functional as F
import tiktoken
from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# 下载模型 
# from transformers import GPT2LMHeadModel
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# print("loaded model from huggingface")

# model.save_pretrained('assets/gpt2')
# print("saved model to assets/gpt2")

# ------------------------------  自动检测GPU  ------------------------------
# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
device = "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
# ------------------------------  输入处理：简易的DataLoader  ------------------------------
# get a data batch
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        # 以B*T为一个batch遍历整个文档，返回x和y(y需要取到下一个token)
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# ------------------------------  模型初始化  ------------------------------
model = GPT(config=GPTConfig())
model.to(device)
train_loader = DataLoaderLite(B=4, T=32)

# ------------------------------  优化器  ------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # type: ignore
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")

import sys; sys.exit(0)

# ------------------------------  评估模型生成文本  ------------------------------
# num_return_sequences = 5
# max_length = 30
# model = GPT.from_pretrained('gpt2', '../../assets/gpt2')
# model.eval()


num_return_sequences = 5
max_length = 30
model = GPT.from_pretrained('gpt2', '../../assets/gpt2')
model.eval()
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello,I'm a-language model,")
tokens = torch.tensor(tokens,dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)#(5, 8)
x = tokens.to(device)
# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # 只获取最后一个token的概率分布
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # 选择概率最大的前50个token，之后的token的概率都设置为0
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # 从50个token中按照概率分布采样一个token
        # multinomial：根据给定的概率分布，返回一个采样的索引
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # 将采样的token的索引从50个token中取出
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # 加入到原来的序列中
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
