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
eval_iters = 200
save_begin = 1000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = f"{vocab_type}_vgg1d_conformer_bigram"

model = NOCTCModel(**model_cfg)
model.to(device)
train_loader = get_loader("train-clean-100", tokenizer, 4, 32)
val_loader = get_loader("dev-clean", tokenizer, 4, 32)
test_loader = get_loader("test-clean", tokenizer, 4, 32)

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
                torch.save(model.state_dict(), f'checkpoints/best_{model_name}.pth')

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

"""emm maybe good?
--- step 17500: val_loss: 0.7125 val_wer: 0.7892 test_loss: 0.7349 test_wer: 0.7890  ---
loss: 1.3168: 100%|█████████████████████████████████████▉| 17839/17840 [1:08:36<00:00,  7.40batch/s]
val_wer: 0.7658862876254181                                                                         
utterances[0]: good by dear randal
transcipts[0]: good five dearre rample
loss: 1.3168: 100%|█████████████████████████████████████▉| 17839/17840 [1:08:50<00:00,  7.40batch/s]
test_wer: 0.7766143106457243                                                                        
utterances[0]: but i didn't know you've only to tell me now
transcipts[0]: i didt know you only tell me now you want
--- step 17839: val_loss: 0.6958 val_wer: 0.7659 test_loss: 0.7161 test_wer: 0.7766  ---
"""