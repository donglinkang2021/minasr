# kaldi.py and transform.py are equivalent. The difference is that kaldi.py does not support batch processing but transform.py does.

import torch
import torch.nn as nn
import torchaudio.functional as F
import torchaudio.compliance.kaldi as K

def cmvn(feature:torch.Tensor, dim_T=0, eps=1e-10):
    return (feature - feature.mean(dim=dim_T, keepdim=True)) / (feature.std(dim=dim_T, keepdim=True) + eps)

class FBANK80(nn.Module):
    def __init__(self, n_mels=80, sample_rate = 16000, frame_length=25, frame_shift=10):
        super().__init__()
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.sample_rate = sample_rate

    def forward(self, waveform):
        fbank = K.fbank(
            waveform=waveform,
            sample_frequency=self.sample_rate,
            num_mel_bins=self.n_mels,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
        )
        return cmvn(fbank, dim_T=1) 
    

class MFCC39(nn.Module):
    def __init__(self, n_mfcc=13, sample_rate = 16000,  frame_length=25, frame_shift=10, use_delta=True):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.use_delta = use_delta
        self.sample_rate = sample_rate

    def forward(self, waveform):
        mfcc = K.mfcc(
            waveform=waveform,
            sample_frequency=self.sample_rate,
            num_ceps=self.n_mfcc,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
        )
        if self.use_delta:
            mfcc = mfcc.unsqueeze(0).transpose(1, 2)
            mfcc_delta1 = F.compute_deltas(mfcc)
            mfcc_delta2 = F.compute_deltas(mfcc_delta1)
            mfcc = torch.cat([mfcc, mfcc_delta1, mfcc_delta2], dim=1)
            mfcc = mfcc.transpose(1, 2).squeeze(0)
        return cmvn(mfcc, dim_T=1)
    