import torch
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from asr.datasets import LibriSpeechDataset
from vocab.tokenizer import Tokenizer
from tqdm import tqdm
from einops import rearrange

import joblib

km_path = "/opt/data/private/linkdom/data/pretrain"

tokenizer = Tokenizer('bpe')
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = LibriSpeechDataset("train-clean-100", tokenizer)

n_clusters = 512
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

n_batches = len(dataset)
print(f"number of batches per epoch: {n_batches}")
pbar = tqdm(total=n_batches, desc="Kmeans training", leave=True, ncols = 200, unit="batch")
for i, (feature, _) in enumerate(dataset):
    feature = feature.cpu().numpy()
    kmeans.partial_fit(feature)
    inertia = - kmeans.score(feature) / len(feature)
    pbar.set_description(f"Processed {i} sample, inertia: {inertia}")
    pbar.update(1)  
pbar.close()

print("Kmeans training finished.")
print("Saving the kmeans model...")
joblib.dump(kmeans, f"{km_path}/kmeans_{n_clusters}")

"""output
Processed 28538 sample, inertia: 23.0962890625: 100%|███████████████████████████████████████████| 28539/28539 [15:30<00:00, 30.66batch/s]
Kmeans training finished.
Saving the kmeans model...
"""