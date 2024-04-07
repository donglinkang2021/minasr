import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import math
import pytorch_lightning as L

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_size, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # project the queries, keys and values
        q, k, v = self.c_attn(x).split(C, dim=2)
        k = rearrange(k, 'B T (nh hs) -> B nh T hs', nh=self.n_head)
        q = rearrange(q, 'B T (nh hs) -> B nh T hs', nh=self.n_head)
        v = rearrange(v, 'B T (nh hs) -> B nh T hs', nh=self.n_head)

        # casual self-attention: ignore "future" keys during attention
        # masked attention
        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, 
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )
        
        # re-assemble all head outputs side by side
        y = rearrange(y, 'B nh T hs -> B T (nh hs)')
        # output projection
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
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.attn = CausalSelfAttention(n_embd, n_head, head_size, block_size, dropout)
        self.mlp = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class GPT(nn.Module):

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.block_size = block_size
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd)
        ))

        self.lm_head = nn.Linear(n_embd, vocab_size)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
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

    def forward(self, idx):
        device = idx.device
        B, T = idx.shape
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (t)
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x
        
    def generate(self, idx, max_new_tokens:int, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits = self(idx_cond)
             # note: using list [-1] to preserve the time dim
            logits = self.lm_head(logits[:, [-1], :])
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class LibriGPT(L.LightningModule):
    def __init__(self, tokenizer, gpt_kwargs:dict, sample_kwargs:dict, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPT(**gpt_kwargs)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, idx):
        return self.model(idx)

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, sync_dist=True, logger=True)
        self.inference()        
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, sync_dist=True, logger=True)
        self.inference()
        return loss
    
    def _common_step(self, batch, batch_idx):
        x, targets = batch
        x = self.model(x)
        logits = self.model.lm_head(x) # (B,T,vocab_size)
        logits = rearrange(logits, 'B T C -> (B T) C')
        targets = rearrange(targets, 'B T -> (B T)')
        loss = self.loss_fn(logits, targets)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
    
    def inference(self):
        context = torch.tensor([self.hparams.tokenizer.spm.bos_id()], dtype=torch.long, device=self.device).unsqueeze(0)
        text = self.hparams.tokenizer.decode(self.model.generate(context, **self.hparams.sample_kwargs)[0].tolist())
        self.logger.experiment.add_text('generated_text', text, self.global_step)
