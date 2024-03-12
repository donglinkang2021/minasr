from pretrain.model import *
from tqdm import tqdm
from asr.datasets import get_loader
from vocab.tokenizer import Tokenizer
from einops import rearrange
import numpy as np
from utils import save_ckpt

tokenizer = Tokenizer('bpe')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

model_dim = 384

features_kwargs = {
    "demo0": [80, "M", 128, "M", model_dim]
}

conformer_kwargs = {
    "input_dim": model_dim,
    "num_heads": 6,
    "ffn_dim": 128,
    "num_layers": 4,
    "depthwise_conv_kernel_size": 31,
    "dropout": 0.2,
}

wav2vec_kwargs = {
    "model_dim": model_dim,
    "features_cfg": features_kwargs["demo0"],
    "conformer_kwargs": conformer_kwargs
}

eval_interval = 100
save_begin = 200
learning_rate = 3e-4
model = Wav2Vec(**wav2vec_kwargs)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
model_name = f"pretrained_vgg1d_conformer"
model_ckpts = save_ckpt(model_name)
print(f"the model checkpoints will be saved at {model_ckpts}.")

train_loader = get_loader("train-clean-100", tokenizer, 4, 32)
val_loader = get_loader("dev-clean", tokenizer, 4, 32)
test_loader = get_loader("test-clean", tokenizer, 4, 32)

@torch.no_grad()
def estimate():
    metrics = {}
    model.eval()
    for split, loader in [('val', val_loader), ('test', test_loader)]:
        losses = []
        for x, y, lx, ly in tqdm(loader, ncols=100, desc=f"Eval {split} processing", leave=False):
            x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
            logits, logits_lengths = model(x, lx)
            targets = logits[:, 1:, :].softmax(dim=-1)
            logits = logits[:, :-1, :]
            logits = rearrange(logits, 'B T C -> (B T) C')
            targets = rearrange(targets, 'B T C-> (B T) C')
            loss = F.cross_entropy(logits, targets)
            losses.append(loss.item())
        metrics[split + '_loss'] = np.mean(losses)

    model.train()
    return metrics

best_loss = 100
n_batches = len(train_loader)
pbar = tqdm(total=n_batches, desc="Processing batches", leave=True, unit="batch")
for iter, (x, y, lx, ly) in enumerate(train_loader):

    if iter % eval_interval == 0 or iter == n_batches - 1:
        metrics = estimate()
        print(f"--- step {iter}:", end=' ')
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}", end=' ')
        print(" ---")

        if iter > save_begin and metrics['val_loss'] < best_loss:
            best_loss = metrics['val_loss']
            torch.save(model.state_dict(), f'{model_ckpts}/best_{model_name}.pth')

    x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
    logits, logits_lengths = model(x, lx)
    targets = logits[:, 1:, :].softmax(dim=-1)
    logits = logits[:, :-1, :]
    logits = rearrange(logits, 'B T C -> (B T) C')
    targets = rearrange(targets, 'B T C-> (B T) C')
    loss = F.cross_entropy(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pbar.set_description(f"loss: {loss.item()}")
    pbar.update(1)

pbar.close()