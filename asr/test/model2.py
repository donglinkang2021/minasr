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
pbar = tqdm(total=len(trainloader), desc="Processing batches", leave=True, unit="batch")

for x, y, lx, ly in trainloader:
    x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
    out, out_lengths = fe_model(x, lx)
    pbar.set_description(f"out.shape: {out.shape} y.shape: {y.shape}")
    pbar.update(1)

pbar.close()


"""
(GPT) root@asr:~/minasr# python dev.py 
out.shape: torch.Size([27, 102, 384]) y.shape: torch.Size([27, 98]): 100%|████████████████████████████████████████████████████████████████████████████████| 892/892 [02:51<00:00,  5.21batch/s]
"""