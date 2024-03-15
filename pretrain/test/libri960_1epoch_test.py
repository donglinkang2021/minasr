from pretrain.datasets import get_libri960_loader
from tqdm import tqdm
import torch
num_workers = 4
batch_size = 64
trainloader = get_libri960_loader(num_workers, batch_size)
device = "cuda" if torch.cuda.is_available() else "cpu"
n_batches = len(trainloader)
print(f"number of batches per epoch: {n_batches}")
pbar = tqdm(total=n_batches, desc="Processing batches", leave=True, unit="batch")
for x, y, lx, ly in trainloader:
    x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
    pbar.set_description(f"x: {x.shape} y: {y.shape} lx: {lx.shape} ly: {ly.shape}")
    pbar.update(1)  
pbar.close()

"""output
(GPT) root@asr:~/minasr# python libri960_1epoch_test.py 
number of batches per epoch: 4395
x: torch.Size([64, 1671, 80]) y: torch.Size([64, 1671]) lx: torch.Size([64]) ly: torch.Size([64]):   2%|▏         | 79/4395 [00:41<35:28,  2.03batch/s]
"""