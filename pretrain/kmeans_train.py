import torch
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from asr.datasets import get_loader
from vocab.tokenizer import Tokenizer
from tqdm import tqdm
from einops import rearrange

import joblib

km_path = "/opt/data/private/linkdom/data/pretrain"

tokenizer = Tokenizer('bpe')
device = "cuda" if torch.cuda.is_available() else "cpu"
loader_kwargs = {
    "tokenizer": tokenizer,
    "batch_size": 64,
    "num_workers": 4
}

trainloader = get_loader("train-clean-100", **loader_kwargs)

n_clusters = 256
init = "k-means++"
batch_size = 10000
tol = 0.0
max_iter = 100
max_no_improvement = 100
n_init = 20
reassignment_ratio = 0.0

kmeans_kwargs = {
    "n_clusters": n_clusters,
    "init": init,
    "batch_size": batch_size,
    "tol": tol,
    "max_iter": max_iter,
    "max_no_improvement": max_no_improvement,
    "n_init": n_init,
    "reassignment_ratio": reassignment_ratio
}

kmeans = MiniBatchKMeans(**kmeans_kwargs)

n_batches = len(trainloader)
print(f"number of batches per epoch: {n_batches}")
print(f"train loader kwargs: {loader_kwargs}")
pbar = tqdm(total=n_batches, desc="Kmeans training", leave=True, ncols = 200, unit="batch")
for i, (x, y, lx, ly) in enumerate(trainloader):
    x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
    x = rearrange(x, 'B T C -> (B T) C')
    x = x.cpu().numpy()
    kmeans.partial_fit(x)

    if i % 10 == 0:
        inertia = - kmeans.score(x) / len(x)
        print(f"---partial fit inertia: {inertia}---")

    pbar.set_description(f"Processed {i} batches")
    pbar.update(1)  
pbar.close()

print("Kmeans training finished.")
print("Saving the kmeans model...")
joblib.dump(kmeans, f"{km_path}/kmeans_{n_clusters}")

"""output
Processed 445 batches: 100%|█████████████████████████████████████████████████████████| 446/446 [02:48<00:00,  2.65batch/s]
Kmeans training finished.
Saving the kmeans model...
"""