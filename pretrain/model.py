import torch
import torch.nn as nn
from torch.nn import functional as F
from torchaudio.models import Conformer
from pretrain.encoder import TransformerEncoder
from pretrain.mask import get_mask
from pretrain.kmeans.codebooks import load_codebooks
from typing import List
from einops import rearrange
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

class BaseModel(nn.Module):
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



class MiniHubert(BaseModel):
    def __init__(self, model_dim, n_class, features_cfg: List, enc_kwargs: dict):
        super().__init__()
        self.feature_extractor = FeatureExtractor(features_cfg)
        self.label_scale = self.feature_extractor.length_scale
        self.post_extract_proj = nn.Linear(features_cfg[-1], model_dim)
        self.dropout_features = nn.Dropout(p=0.1, inplace=False)
        self.final_proj = nn.Linear(model_dim, features_cfg[0])
        self.final_ln = nn.LayerNorm(features_cfg[0])
        self.codebooks = load_codebooks(n_class).requires_grad_(False).to(device)
        self.apply(self._init_weights)
        self.encoder = TransformerEncoder(**enc_kwargs)
        print(f"number of parameters of whole model: {self.get_num_params()/1e6:.6f} M ")

    def align_labels(self, T:int, pesudo_label:torch.LongTensor):
        """ Subsample pesudo_label to match the length of the output of the model """
        target_inds = torch.arange(0, pesudo_label.size(1), self.label_scale)
        targets = pesudo_label[:, target_inds]
        targets = targets[:, :T]
        return targets
    
    def attention(self, logits):
        attention = logits @ self.codebooks.T / math.sqrt(logits.size(-1))
        return attention
    
    def forward(self, x, x_lengths, pesudo_label=None):
        x, x_lengths = self.feature_extractor(x, x_lengths)
        x = self.post_extract_proj(x)
        x = self.dropout_features(x)
        x = get_mask(x)
        x = self.encoder(x)
        logits = self.final_proj(x) # (B,T,features_cfg[0])
        if pesudo_label is not None:
            logits = self.final_ln(logits)
            logits = self.attention(logits) # F.cosine_similarity will let the GPU memory explode
            targets = self.align_labels(logits.size(1), pesudo_label) # (B, T)
            logits = rearrange(logits, 'B T C -> (B T) C')
            targets = rearrange(targets, 'B T -> (B T)')
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, x_lengths, loss

    def freeze(self):
        # freeze the model parameters except the final projection layer
        for param in self.parameters():
            param.requires_grad = False
        for param in self.final_proj.parameters():
            param.requires_grad = True
    
class LinearHead(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(model_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.proj(x)