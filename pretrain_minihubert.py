from pretrain.model import MiniHubert
import torch
import joblib
from asr.datasets import get_loader
from vocab.tokenizer import Tokenizer
from tqdm import tqdm
import numpy as np
from utils import save_ckpt

vocab_type = 'bpe'
tokenizer = Tokenizer(vocab_type)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

model_dim = 768
n_clusters = 256
features_kwargs = {
    "demo0": [80, "M", 128, "M", 256, 256, "M", 512]
}

device = "cuda" if torch.cuda.is_available() else "cpu"
loader_kwargs = {
    "tokenizer": tokenizer,
    "batch_size": 64,
    "num_workers": 4
}

enc_kwargs = {
    "n_embd": model_dim,
    "n_head": 6,
    "n_layer": 12,
    "block_size": 2048,
    "dropout": 0.1
}

model_cfg = {
    "model_dim": model_dim,
    "n_class": n_clusters,
    "features_cfg": features_kwargs["demo0"],
    "enc_kwargs": enc_kwargs
}


km_path = "/opt/data/private/linkdom/data/pretrain/"
km_model = joblib.load(f"{km_path}/kmeans_{n_clusters}")
codebooks = km_model.cluster_centers_
codebooks = torch.from_numpy(codebooks).to(device)
print(f"Kmeans model loaded from {km_path}, codebooks shape: {codebooks.shape}")

@torch.no_grad()
def kmeans_label(x: torch.Tensor, codebooks: torch.Tensor) -> torch.LongTensor:
    return torch.cdist(x, codebooks).argmin(dim=-1)

# train config
num_epochs = 20
eval_interval = 500
save_begin = 1000
learning_rate = 3e-4
model_name = f"pretrain_minihubert_kmeans_{n_clusters}"
model_ckpts = save_ckpt(model_name)
print(f"the model checkpoints will be saved at {model_ckpts}.")

model = MiniHubert(**model_cfg).to(device)
prev_ckpt = 'checkpoints/pretrain_minihubert_kmeans_256/2024-03-11_18:47:49/best_pretrain_minihubert_kmeans_256.pth'
model.load_state_dict(torch.load(prev_ckpt))
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

trainloader = get_loader("train-clean-360", **loader_kwargs)
valloader = get_loader("dev-clean", **loader_kwargs)
testloader = get_loader("test-clean", **loader_kwargs)


@torch.no_grad()
def estimate():
    metrics = {}
    model.eval()
    for split, loader in [('val', valloader), ('test', testloader)]:
        losses = []
        for x, _, lx, _ in tqdm(loader, ncols=100, desc=f"Eval {split} processing", leave=False):
            x, lx = x.to(device), lx.to(device)
            pesudo_label = kmeans_label(x, codebooks)
            _, _, loss = model(x, lx, pesudo_label)
            losses.append(loss.item())
        metrics[split + '_loss'] = np.mean(losses)
    model.train()
    return metrics

best_loss = 100
n_batches = len(trainloader)
print(f"number of batches per epoch: {n_batches}")
print(f"train loader kwargs: {loader_kwargs}")
pbar = tqdm(total=num_epochs * n_batches, desc="model pretraining", leave=True, unit="batch")
for epoch in range(num_epochs):
    for i, (x, _, lx, _) in enumerate(trainloader):
        iter = epoch * n_batches + i
        if iter % eval_interval == 0 or iter == n_batches - 1:
            metrics = estimate()
            print(f"\n--- step {iter}: {metrics} ---")
            if iter > save_begin and metrics['val_loss'] < best_loss:
                best_loss = metrics['val_loss']
                torch.save(model.state_dict(), f'{model_ckpts}/best_{model_name}.pth')
        x, lx = x.to(device), lx.to(device)
        pesudo_label = kmeans_label(x, codebooks)
        _, _, loss = model(x, lx, pesudo_label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"model pretraining, loss: {loss.item():.4f}")
        pbar.update(1)  
pbar.close()

"""
model pretraining, loss: 3.4960: 100%|██████████| 32520/32520 [3:54:05<00:00,  2.32batch/s]
--- step 32500: {'val_loss': 1.4669770944950193, 'test_loss': 1.4505024348817221} ---
"""