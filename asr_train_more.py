import torch
import torch.nn.functional as F
from tqdm import tqdm
from asr.datasets import get_loader
from asr.model import NOCTCModel
from vocab.tokenizer import Tokenizer
from einops import rearrange, repeat
import numpy as np
from jiwer import wer
from pathlib import Path
from datetime import datetime
from utils import save_ckpt

vocab_type = 'bpe'
tokenizer = Tokenizer(vocab_type)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

model_dim = 384

features_kwargs = {
    "demo0": [80, "M", 128, "M", 256, "M", model_dim, "M"]
}

conformer_kwargs = {
    "input_dim": model_dim,
    "num_heads": 6,
    "ffn_dim": 128,
    "num_layers": 4,
    "depthwise_conv_kernel_size": 31,
    "dropout": 0.2,
}

gpt_kwargs = {
    "vocab_size": tokenizer.vocab_size,
    "n_embd": model_dim,
    "n_head": 6,
    "n_layer": 8,
    "block_size": 512,
    "dropout": 0.2
}

model_cfg = {
    "features_cfg": features_kwargs["demo0"],
    "conformer_kwargs": conformer_kwargs,
    "gpt_kwargs": gpt_kwargs,
    "is_lm_pretrained": False
}

# sample config
sample_kwargs = {
    "max_new_tokens": 100, # max number of tokens to generate
    "temperature": 0.8, # > 1.0 = more exploratory, < 1.0 = more conservative
    "top_k": 200 # consider only the top_k most likely tokens, clamp others to have 0 probability
}

# train config
num_epochs = 20
eval_interval = 500
save_begin = 1000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = f"{vocab_type}_vgg1d_conformer_bigram"
model_ckpts = save_ckpt(model_name)
print(f"the model checkpoints will be saved at {model_ckpts}.")

model = NOCTCModel(**model_cfg)
model.to(device)
prev_ckpt = 'checkpoints/bpe_vgg1d_conformer_bigram/2024-02-25_22:50:35/best_bpe_vgg1d_conformer_bigram.pth'
model.load_state_dict(torch.load(prev_ckpt))
train_loader = get_loader("train-clean-100", tokenizer, 4, 32) # more than 1 hour trainning
# train_loader = get_loader("train-clean-360", tokenizer, 4, 32) # about 5 hour trainning
# train_loader = get_loader("train-other-500", tokenizer, 4, 32) # about 7 hour trainning
val_loader = get_loader("dev-clean", tokenizer, 4, 32)
test_loader = get_loader("test-clean", tokenizer, 4, 32)

criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate():
    metrics = {}
    model.eval()
    for split, loader in [('val', val_loader), ('test', test_loader)]:
        losses = []
        for x, y, lx, ly in tqdm(loader, ncols=100, desc=f"Eval {split} processing", leave=False):
            x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
            idx = y[:, :-1]
            targets = y[:, 1:]
            logits = model(x, lx, idx)
            logits = rearrange(logits, 'B T C -> (B T) C')
            targets = rearrange(targets, 'B T -> (B T)')
            loss = F.cross_entropy(logits, targets)
            losses.append(loss.item())
        metrics[split + '_loss'] = np.mean(losses)

        # calculate wer of the last batch
        B = x.size(0)
        context = torch.tensor([tokenizer.spm.bos_id()], dtype=torch.long, device=device)
        context = repeat(context, 'n -> b n', b=B)
        transcipts = tokenizer.decode(model.transcribe(x, lx, context, **sample_kwargs).tolist())
        utterances = tokenizer.decode(y.tolist())
        metrics[split + '_wer'] = wer(utterances, transcipts)
        print(f"\n{split}_wer:", metrics[split + '_wer'])
        print("utterances[0]:", utterances[0])
        print("transcipts[0]:", transcipts[0])

    model.train()
    return metrics

best_loss = 100
n_batches = len(train_loader)
pbar = tqdm(total=num_epochs * n_batches, ncols=100, desc="Training ASR", leave=False, unit="batch")
for epoch in range(num_epochs):
    for i, (x, y, lx, ly) in enumerate(train_loader):

        iter = epoch * n_batches + i
        if iter % eval_interval == 0 or iter == num_epochs * n_batches - 1:
            metrics = estimate()
            print(f"--- step {iter}:", end=' ')
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}", end=' ')
            print(" ---")

            if iter > save_begin and metrics['val_loss'] < best_loss:
                best_loss = metrics['val_loss']
                torch.save(model.state_dict(), f'{model_ckpts}/best_{model_name}.pth')

        x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
        idx = y[:, :-1]
        targets = y[:, 1:]
        logits = model(x, lx, idx)
        logits = rearrange(logits, 'B T C -> (B T) C')
        targets = rearrange(targets, 'B T -> (B T)')
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss: {loss.item():.4f}")
        pbar.update(1)
pbar.close()