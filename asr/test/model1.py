from asr.model import *
from tqdm import tqdm
from asr.datasets import get_loader
from vocab.tokenizer import Tokenizer

tokenizer = Tokenizer('bpe')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

features_kwargs = {
    "demo0": [80, "M", 128, "M", 256, 256, "M", 384, "M"]
}

fe_model = FeatureExtractor(features_kwargs["demo0"])
fe_model.to(device)

trainloader = get_loader("train-clean-100", tokenizer, 4, 32)

x, y, lx, ly = next(iter(trainloader))
x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)

print(f"x.shape: {x.shape} y.shape: {y.shape} lx.shape: {lx.shape} ly.shape: {ly.shape}")

with torch.no_grad():
    out, out_lengths = fe_model(x, lx)
    print(f"out.shape: {out.shape} out_lengths.shape: {out_lengths.shape}")

"""
(GPT) root@asr:~/minasr# python dev.py 
x.shape: torch.Size([32, 1631, 80]) y.shape: torch.Size([32, 85]) lx.shape: torch.Size([32]) ly.shape: torch.Size([32])
out.shape: torch.Size([32, 101, 384]) out_lengths.shape: torch.Size([32])
"""