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
    pbar.set_description(f"x.shape: {x.shape} y.shape: {y.shape} lx.shape: {lx.shape} ly.shape: {ly.shape}")
    pbar.update(1)  
pbar.close()
