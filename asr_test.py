import torch
import torch.nn.functional as F
from tqdm import tqdm
from asr.datasets import get_loader
from asr.model import NOCTCModel
from vocab.tokenizer import Tokenizer
from einops import rearrange, repeat
import numpy as np
from jiwer import wer

vocab_type = 'bpe'
tokenizer = Tokenizer(vocab_type)
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

# sample config
sample_kwargs = {
    "max_new_tokens": 200, # max number of tokens to generate
    "temperature": 0.8, # > 1.0 = more exploratory, < 1.0 = more conservative
    "top_k": 200 # consider only the top_k most likely tokens, clamp others to have 0 probability
}

# train config
num_epochs = 20
eval_interval = 500
eval_iters = 200
save_begin = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = f"{vocab_type}_vgg1d_conformer_bigram"

model = NOCTCModel(**model_cfg)
model.to(device)
model.load_state_dict(torch.load(f'checkpoints/best_{model_name}.pth'))
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
        wer_scores = []
        for x, y, lx, ly in tqdm(loader, ncols=100, desc=f"Eval {split} processing", leave=False):
            x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
            idx = y[:, :-1]
            targets = y[:, 1:]
            logits = model(x, lx, idx)
            logits = rearrange(logits, 'B T C -> (B T) C')
            targets = rearrange(targets, 'B T -> (B T)')
            loss = F.cross_entropy(logits, targets)
            losses.append(loss.item())

            # calculate wer
            B = x.size(0)
            context = torch.tensor([tokenizer.spm.bos_id()], dtype=torch.long, device=device)
            context = repeat(context, 'n -> b n', b=B)
            transcipts = tokenizer.decode(model.transcribe(x, lx, context, **sample_kwargs).tolist())
            utterances = tokenizer.decode(y.tolist())
            wer_scores.append(wer(utterances, transcipts))

        metrics[split + '_loss'] = np.mean(losses)
        metrics[split + '_wer'] = np.mean(wer_scores)

    model.train()
    return metrics


metrics = estimate()
print(f"--- eval:", end=' ')
for k, v in metrics.items():
    print(f"{k}: {v:.4f}", end=' ')
print(" ---")
