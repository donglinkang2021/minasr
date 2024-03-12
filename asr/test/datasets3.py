from asr.datasets import *
from vocab.tokenizer import Tokenizer
from tqdm import tqdm

tokenizer = Tokenizer('bpe')
device = "cuda" if torch.cuda.is_available() else "cpu"
loader_kwargs = {
    "tokenizer": tokenizer,
    "batch_size": 64,
    "num_workers": 4
}

trainloader = get_loader("train-clean-100", **loader_kwargs)
valloader = get_loader("dev-clean", **loader_kwargs)
testloader = get_loader("test-clean", **loader_kwargs)

n_batches = len(trainloader)
print(f"number of batches per epoch: {n_batches}")
print(f"train loader kwargs: {loader_kwargs}")

# create a progress bar
pbar = tqdm(total=n_batches, desc="Processing batches", leave=True, unit="batch")
for x, y, lx, ly in trainloader:
    x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
    pbar.set_description(f"x.shape: {x.shape} y.shape: {y.shape} lx.shape: {lx.shape} ly.shape: {ly.shape}")
    pbar.update(1)  
pbar.close()

"""output
(GPT) root@asr:~/minasr# python datasets3.py 
number of batches per epoch: 446
train loader kwargs: {'tokenizer': <Tokenizer vocab_size=1000 token_type=bpe>, 'batch_size': 64, 'num_workers': 4}
x.shape: torch.Size([59, 1625, 80]) y.shape: torch.Size([59, 96]) lx.shape: torch.Size([59]) ly.shape: torch.Size([59]): 100%|█████████████████████████████████████████████| 446/446 [05:51<00:00,  1.27batch/s]
"""
