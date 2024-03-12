import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import math

class MultiheadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.dropout = dropout
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        k = rearrange(k, 'B T (nh hs) -> B nh T hs', nh=self.n_head)
        q = rearrange(q, 'B T (nh hs) -> B nh T hs', nh=self.n_head)
        v = rearrange(v, 'B T (nh hs) -> B nh T hs', nh=self.n_head)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, 
            dropout_p=self.dropout if self.training else 0,
        )
        
        y = rearrange(y, 'B nh T hs -> B T (nh hs)')
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias=True):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.attn = MultiheadAttention(n_embd, n_head, dropout)
        self.mlp = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        assert block_size is not None
        self.block_size = block_size
        
        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            layers = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd)
        ))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        device = x.device
        B, T, n_embd = x.shape
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (t)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        x = self.transformer.drop(x + pos_emb)
        for block in self.transformer.layers:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x