import torch
import torch.nn as nn
from torch.nn import functional as F
from torchaudio.models import Conformer
from lm.model import GPT
from typing import List
from einops import rearrange

class ASRModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_lengths):
        raise NotImplementedError
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class FeatureExtractor(nn.Module):
    def __init__(self, features_cfg: List):
        super().__init__()
        self.length_scale = self.get_length_scale(features_cfg)
        self.features = self._make_layers(features_cfg, batch_norm=True)

    def get_length_scale(self, features_cfg):
        subsample_num = len([x for x in features_cfg if x == "M"])
        return 2**subsample_num

    def forward(self, x, x_lengths):
        x = x.transpose(1, 2) # BxTxC -> BxCxT
        x = self.features(x)
        x = x.transpose(1, 2) # BxCxT -> BxTxC
        x_lengths = x_lengths // self.length_scale
        return x, x_lengths

    def _make_layers(self, cfg: List, batch_norm: bool = False):
        layers: List[nn.Module] = []
        in_channels = cfg[0]
        for layer_type in cfg:
            if layer_type == "M":
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv1d(in_channels, layer_type, kernel_size=3, padding=1))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_type))
                layers.append(nn.ReLU(inplace=True))
                in_channels = layer_type
        return nn.Sequential(*layers)

class NOCTCModel(ASRModel):
    def __init__(
        self, 
        features_cfg: List, 
        conformer_kwargs: dict, 
        gpt_kwargs: dict, 
        is_lm_pretrained: bool = True, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.features = FeatureExtractor(features_cfg)
        self.conformer = Conformer(**conformer_kwargs)
        self.lm = GPT(**gpt_kwargs)
        self.n_embd = gpt_kwargs["n_embd"]
        self.n_head = gpt_kwargs["n_head"]
        self.kv_proj = nn.Linear(self.n_embd, 2 * self.n_embd, bias=False)
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.dropout = dropout
        self.apply(self._init_weights)
        if is_lm_pretrained:
            self.load_lm()
        print(f"number of parameters of asr model: {self.get_num_params()/1e6:.6f} M ")

    def load_lm(self):
        self.lm.load_state_dict(torch.load("checkpoints/best_bpe_gpt.pth"))

    def freeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = False

    def lm_forward(self, idx):
        device = idx.device
        B, T = idx.shape
        assert T <= self.lm.block_size, f"Cannot forward sequence of length {T}, block size is only {self.lm.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.lm.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        pos_emb = self.lm.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        x = self.lm.transformer.drop(tok_emb + pos_emb)
        for block in self.lm.transformer.h:
            x = block(x)
        x = self.lm.transformer.ln_f(x) # BxTxembd
        return x

    def forward(self, x, x_lengths, idx):
        # encoder
        # we: waveform embeddings
        we, we_lengths = self.features(x, x_lengths) # BxT0xfbank -> BxT1xC
        we, we_lengths = self.conformer(we, we_lengths) # BxT1xC -> BxT1xC

        # decoder
        # te: token embeddings
        te = self.lm_forward(idx)

        # cross attention
        B, T, C = we.shape
        k, v = self.kv_proj(we).split(C, dim=2) # BxT1x2C -> BxT1xC, BxT1xC
        q = self.q_proj(te) # BxT2xC
        q = rearrange(q, 'B T2 (nh hs) -> B nh T2 hs', nh=self.n_head)
        k = rearrange(k, 'B T1 (nh hs) -> B nh T1 hs', nh=self.n_head)
        v = rearrange(v, 'B T1 (nh hs) -> B nh T1 hs', nh=self.n_head)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, 
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )

        y = rearrange(y, 'B nh T2 hs -> B T2 (nh hs)')
        y = self.lm.lm_head(y) # BxT2xV
        return y
    
    def transcribe(self, x, x_lengths, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.lm.block_size:]
            # get the predictions
            logits = self(x, x_lengths, idx_cond)
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
    
class ConformerCTC(ASRModel):
    def __init__(self, features_cfg: List, conformer_kwargs: dict, vocab_size: int):
        super().__init__()
        model_dim = conformer_kwargs["input_dim"]
        features_dim = features_cfg[-1]
        self.features = FeatureExtractor(features_cfg)
        self.post_features = nn.Linear(features_dim, model_dim)
        self.conformer = Conformer(**conformer_kwargs)
        self.head = nn.Linear(model_dim, vocab_size)
        # self.apply(self._init_weights)
    
    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor):
        x, x_lengths = self.features(x, x_lengths)
        x = self.post_features(x)
        x, x_lengths = self.conformer(x, x_lengths)
        x = self.head(x)
        return x, x_lengths