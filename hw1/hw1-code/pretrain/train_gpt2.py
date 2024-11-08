

import math
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

# ------------------------------  torch相关初始化  ------------------------------
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
# 调整类型为 TF32
torch.set_float32_matmul_precision('high')
# ------------------------------  模型初始化  ------------------------------
model = GPT(config=GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model)

# ------------------------------  数据加载  ------------------------------
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 8 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)



# ------------------------------  优化器  ------------------------------
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it: int) -> float:
    # 1) 线性warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) lr > max_steps时，返回min_lr
    if it > max_steps:
        return min_lr
    # 3) 介于两者之间时，使用余弦衰减到min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# ------------------------------  训练模型  ------------------------------
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    # 调整精度为bfloat16
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # 必须缩放损失以考虑梯度累积，因为梯度只是在每次backward()中添加的。
        # 梯度的添加对应于目标中的SUM，但我们想要MEAN。在这里缩放损失，达到正确结果
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # 确定本次迭代的学习率
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:4d} | loss: {loss_accum:.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
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
