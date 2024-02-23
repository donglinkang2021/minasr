import torch
import torch.nn.functional as F
from tqdm import tqdm
from asr.datasets import get_loader
from asr.model import NOCTCModel
from vocab.tokenizer import Tokenizer
from einops import rearrange

tokenizer = Tokenizer('bpe')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

features_kwargs = {
    "demo0": [80, "M", 128, "M", 256, 256, "M", 384, "M"]
}

conformer_kwargs = {
    "input_dim": 384,
    "num_heads": 4,
    "ffn_dim": 128,
    "num_layers": 6,
    "depthwise_conv_kernel_size": 31,
    "dropout": 0.2,
}

gpt_kwargs = {
    "vocab_size": tokenizer.vocab_size,
    "n_embd": 384,
    "n_head": 6,
    "n_layer": 6,
    "block_size": 512,
    "dropout": 0.2
}

model_cfg = {
    "features_cfg": features_kwargs["demo0"],
    "conformer_kwargs": conformer_kwargs,
    "gpt_kwargs": gpt_kwargs,
    "is_lm_pretrained": True
}

learning_rate = 3e-4

model = NOCTCModel(**model_cfg)
model.to(device)
trainloader = get_loader("train-clean-100", tokenizer, 4, 32)
criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

pbar = tqdm(total=len(trainloader), desc="Demo training", leave=True, unit="batch")

for x, y, lx, ly in trainloader:
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
