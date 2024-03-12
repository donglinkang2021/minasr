import torch
import joblib
from asr.datasets import get_loader
from vocab.tokenizer import Tokenizer
from tqdm import tqdm

tokenizer = Tokenizer('bpe')
device = "cuda" if torch.cuda.is_available() else "cpu"
loader_kwargs = {
    "tokenizer": tokenizer,
    "batch_size": 64,
    "num_workers": 4
}

n_clusters = 256
km_path = "/opt/data/private/linkdom/data/pretrain/"
km_model = joblib.load(f"{km_path}/kmeans_{n_clusters}")
codebooks = km_model.cluster_centers_
codebooks = torch.from_numpy(codebooks).to(device)
print(f"Kmeans model loaded from {km_path}, codebooks shape: {codebooks.shape}")

@torch.no_grad()
def kmeans_label(x: torch.Tensor, codebooks: torch.Tensor) -> torch.LongTensor:
    return torch.cdist(x, codebooks).argmin(dim=-1)

trainloader = get_loader("train-clean-100", **loader_kwargs)
n_batches = len(trainloader)
print(f"number of batches per epoch: {n_batches}")
print(f"train loader kwargs: {loader_kwargs}")
pbar = tqdm(total=n_batches, desc="Kmeans training", leave=True, ncols = 200, unit="batch")
for i, (x, y, lx, ly) in enumerate(trainloader):
    x = x.to(device)
    pesudo_label = kmeans_label(x, codebooks)
    pbar.set_description(f"Processed {i} batches")
    pbar.update(1)  
pbar.close()
