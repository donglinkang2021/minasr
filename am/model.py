import torch
import torch.nn as nn
from torch.nn import functional as F
from torchaudio.models import Conformer
from typing import List

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

class Wav2Vec(BaseModel):
    def __init__(self, model_dim, features_cfg: List, conformer_kwargs: dict):
        super().__init__()
        self.feature_extractor = FeatureExtractor(features_cfg)
        self.conformer = Conformer(**conformer_kwargs)
        # self.apply(self._init_weights)
        print(f"number of parameters of asr model: {self.get_num_params()/1e6:.6f} M ")
    
    def forward(self, x, x_lengths):
        x, x_lengths = self.feature_extractor(x, x_lengths)
        x, x_lengths = self.conformer(x, x_lengths)
        return x, x_lengths