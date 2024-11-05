

import time
import torch
from torch.nn import functional as F
import tiktoken
from model import GPT, GPTConfig
from dataloader import DataLoaderLite

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

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# ------------------------------  模型初始化  ------------------------------
model = GPT(config=GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model)
train_loader = DataLoaderLite(B=8, T=1024)

# 调整类型为 TF32
torch.set_float32_matmul_precision('high')

# ------------------------------  优化器  ------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # type: ignore
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # 调整精度为bfloat16
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        # import code; code.interact(local=locals())
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T
    tokens_per_sec = tokens_processed / dt
    print(f"step {i:4d} | loss: {loss.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

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
