import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lm.model import GPT
from vocab.tokenizer import Tokenizer
from lm.datasets import get_loader
from tqdm import tqdm

vocab_type = 'bpe'
tokenizer = Tokenizer(vocab_type)

gpt_kwargs = {
    "vocab_size": tokenizer.vocab_size,
    "n_embd": 384,
    "n_head": 6,
    "n_layer": 6,
    "block_size": 512,
    "dropout": 0.2
}

# train config
num_epochs = 10
eval_interval = 500
eval_iters = 200
save_begin = 20000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = f"{vocab_type}_gpt"

loader_kwargs = {
    "tokenizer": tokenizer,
    "batch_size": 64,
    "num_workers": 2
}

# sample config
sample_kwargs = {
    "max_new_tokens": 200, # max number of tokens to generate
    "temperature": 0.8, # > 1.0 = more exploratory, < 1.0 = more conservative
    "top_k": 200 # consider only the top_k most likely tokens, clamp others to have 0 probability
}

# ------------
torch.manual_seed(1337)

model = GPT(**gpt_kwargs)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_loader = get_loader(split='train', **loader_kwargs)
val_loader = get_loader(split='dev', **loader_kwargs)
test_loader = get_loader(split='test', **loader_kwargs)

@torch.no_grad()
def estimate():
    metrics = {}
    model.eval()
    for split, loader in [('val', val_loader), ('test', test_loader)]:
        losses = []
        num_samples = 0
        for x, y in tqdm(loader, ncols=100, desc=f"Eval {split} processing", leave=False):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
            num_samples += x.shape[0]
        metrics[split + '_loss'] = np.mean(losses)
    model.train()
    return metrics

best_loss = 100
n_batches = len(train_loader)
pbar = tqdm(total=num_epochs * n_batches, ncols=100, desc="Training GPT", leave=False, unit="batch")
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):

        iter = epoch * n_batches + i
        if iter % eval_interval == 0 or iter == num_epochs * n_batches - 1:
            metrics = estimate()
            print(f"--- step {iter}:", end=' ')
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}", end=' ')
            print(" ---")

            if iter > save_begin and metrics['val_loss'] < best_loss:
                best_loss = metrics['val_loss']
                torch.save(model.state_dict(), f'checkpoints/best_{model_name}.pth')

        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss: {loss.item():.4f}")
        pbar.update(1)
pbar.close()

text = "hello"
context = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device).unsqueeze(0)
print(tokenizer.decode(model.generate(context, **sample_kwargs)[0].tolist()))