

from dataclasses import dataclass
import math
import torch
from torch import  nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

# ========================= GPT Model =========================
class CausalSelfAttention(nn.Module):
    '''
    多头自注意力机制
    '''
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # 保证embedding维度可以被head数整除
        # c_attn：全连接层，将一个batch中输入的embedding映射到3个部分，分别是key, query, value
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # c_proj：全连接层，将多头注意力机制的输出映射到原始的embedding维度，用作输出
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        # 生成一个下三角矩阵，用于mask掉未来的信息
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
 
    def forward(self, x: torch.Tensor):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size() 
        # 对一个batch的所有头计算query, key, value
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # nh 表示头的数量，hs表示头的大小，C表示通道数= nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        # 将query、key和value重塑并转置，以适应多头自注意力的计算。
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # 计算注意力分数，通过query和key的转置的矩阵乘法，然后除以key维度的平方根。
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 使用之前定义的下三角矩阵bias来mask未来的信息，将这些位置的注意力分数设置为负无穷。
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # 计算softmax得到注意力权重
        att = F.softmax(att, dim=-1)
        # 使用注意力权重和value计算加权的value，得到自注意力的输出。
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # 将多头自注意力的输出重新组合在一起，并调整形状以匹配原始输入的维度。
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    '''
    全连接层，GPT2中的MLP是一个两层的全连接网络，中间有一个GELU激活函数
    形状变化：[batch_size, seq_len, n_embd] -> [batch_size, seq_len, 4 * n_embd] -> [batch_size, seq_len, n_embd]
    '''
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh') # 使用tanh近似的GELU激活函数（GPT2使用，现在没必要用）
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    '''
    transformer block
    '''
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        # 和原transformer不同的是，使用一个干净的残差连接，而不是残差中间经过一个Norm
        x = x + self.attn(self.ln_1(x)) # 注意力是聚合、池化、加权求和的过程
        x = x + self.mlp(self.ln_2(x)) # MLP是发生在每个单独的token上，token之间没有联系，是mapping操作 
        return x
    


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional encoding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # 隐藏层权重
            ln_f = nn.LayerNorm(config.n_embd), # layer normalization
        ))
        # 最终的输出层
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

