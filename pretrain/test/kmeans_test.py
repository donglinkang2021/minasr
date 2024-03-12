import torch
import joblib
from asr.datasets import get_loader
from vocab.tokenizer import Tokenizer

n_clusters = 256
vocab_type = 'bpe'
tokenizer = Tokenizer(vocab_type)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
loader_kwargs = {
    "tokenizer": tokenizer,
    "batch_size": 64,
    "num_workers": 4
}
trainloader = get_loader("train-clean-360", **loader_kwargs)

@torch.no_grad()
def kmeans_label(x: torch.Tensor, codebooks: torch.Tensor) -> torch.LongTensor:
    return torch.cdist(x, codebooks).argmin(dim=-1)

km_path = "/opt/data/private/linkdom/data/pretrain/"
km_model = joblib.load(f"{km_path}/kmeans_{n_clusters}")
codebooks = km_model.cluster_centers_
codebooks = torch.from_numpy(codebooks).to(device)
print(f"Kmeans model loaded from {km_path}, codebooks shape: {codebooks.shape}")

best_loss = 100
n_batches = len(trainloader)
print(f"number of batches per epoch: {n_batches}")
print(f"train loader kwargs: {loader_kwargs}")
for i, (x, _, lx, _) in enumerate(trainloader):
    x, lx = x.to(device), lx.to(device)
    pesudo_label = kmeans_label(x, codebooks)
    print(f"batch {i+1}/{n_batches}, pesudo_label shape: {pesudo_label.shape}")
    print(f"pesudo_label[0][:100]: {pesudo_label[0][:100]}")
    break