from dataclasses import dataclass
import inspect
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
        self.c_proj.NANOGPT_SCALE_INIT = 1 # type: ignore
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

        # # 计算注意力分数，通过query和key的转置的矩阵乘法，然后除以key维度的平方根。
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # # 使用之前定义的下三角矩阵bias来mask未来的信息，将这些位置的注意力分数设置为负无穷。
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # # 计算softmax得到注意力权重
        # att = F.softmax(att, dim=-1)
        # # 使用注意力权重和value计算加权的value，得到自注意力的输出。
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
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
        self.c_proj.NANOGPT_SCALE_INIT = 1 # type: ignore
        
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

        # 共享权重：将token embedding的权重和最终的输出层的权重共享
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化权重
        # self.apply函数会递归地将函数应用到模型的所有模块上
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 线性层采用正态分布初始化，均值为0，标准差为0.02，偏置为0
        # 残差连接中为x + sublayer(x)，每个残差块都有贡献，多次累加导致其方差变大，使用缩放因子进行缩放 ：
        # 嵌入层采用正态分布初始化，均值为0，标准差为0.02
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx: torch.Tensor, targets: torch.Tensor|None = None):
        # idx:输入序列的token索引 (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        # 将token嵌入向量和位置嵌入向量相加，得到最终的嵌入表示。
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        # 得到一个用对数表示的概率分布，表示下一个token的概率。
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # 计算交叉熵损失，将向量展平为2D张量(B * T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type:str, model_path:str|None=None):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_path or model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # type: ignore
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused) # type: ignore
        return optimizer