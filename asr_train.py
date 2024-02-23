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
    "max_new_tokens": 100, # max number of tokens to generate
    "temperature": 0.8, # > 1.0 = more exploratory, < 1.0 = more conservative
    "top_k": 200 # consider only the top_k most likely tokens, clamp others to have 0 probability
}

# train config
num_epochs = 20
eval_interval = 500
eval_iters = 200
save_begin = 1000
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = f"{vocab_type}_vgg1d_conformer_bigram"

model = NOCTCModel(**model_cfg)
model.to(device)
model.freeze_lm()
train_loader = get_loader("train-clean-100", tokenizer, 4, 32)
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

"""
(GPT) root@asr:~/minasr# python dev.py
number of parameters: 11.409640 M 
number of parameters: 20.188120 M 
Training ASR:   0%|                                                     | 0/8920 [00:00<?, ?batch/s]--- step 0: val_loss: 7.4000 test_loss: 7.3844  ---
loss: 2.8292:   6%|██▍                                        | 499/8920 [01:47<28:33,  4.91batch/s]--- step 500: val_loss: 1.4237 test_loss: 1.3708  ---
loss: 2.5109:  11%|████▊                                      | 999/8920 [03:39<23:45,  5.56batch/s]--- step 1000: val_loss: 1.3364 test_loss: 1.3139  ---
loss: 2.6908:  17%|███████                                   | 1499/8920 [05:33<23:01,  5.37batch/s]--- step 1500: val_loss: 1.2951 test_loss: 1.2749  ---
loss: 2.3490:  22%|█████████▍                                | 2000/8920 [07:40<17:29,  6.60batch/s]--- step 2000: val_loss: 1.3089 test_loss: 1.2814  ---
loss: 2.2990:  28%|███████████▊                              | 2500/8920 [09:18<21:55,  4.88batch/s]--- step 2500: val_loss: 1.2952 test_loss: 1.2528  ---
loss: 2.4389:  34%|██████████████                            | 2999/8920 [11:05<19:44,  5.00batch/s]--- step 3000: val_loss: 1.2653 test_loss: 1.2603  ---
loss: 2.5782:  39%|████████████████▍                         | 3500/8920 [13:10<13:44,  6.57batch/s]--- step 3500: val_loss: 1.2409 test_loss: 1.2339  ---
loss: 2.1946:  45%|██████████████████▊                       | 3999/8920 [14:52<14:53,  5.51batch/s]--- step 4000: val_loss: 1.2441 test_loss: 1.2075  ---
loss: 2.2219:  50%|█████████████████████▏                    | 4500/8920 [17:00<10:07,  7.27batch/s]--- step 4500: val_loss: 1.2573 test_loss: 1.1906  ---
loss: 2.1994:  56%|███████████████████████▌                  | 5000/8920 [18:39<14:41,  4.45batch/s]--- step 5000: val_loss: 1.2183 test_loss: 1.1743  ---
loss: 2.2461:  62%|█████████████████████████▉                | 5499/8920 [20:34<10:49,  5.26batch/s]--- step 5500: val_loss: 1.1918 test_loss: 1.1445  ---
loss: 2.1137:  67%|████████████████████████████▏             | 5999/8920 [22:24<09:44,  5.00batch/s]--- step 6000: val_loss: 1.1896 test_loss: 1.1639  ---
loss: 2.0682:  73%|██████████████████████████████▌           | 6500/8920 [24:30<05:57,  6.78batch/s]--- step 6500: val_loss: 1.1832 test_loss: 1.1535  ---
loss: 2.1188:  78%|████████████████████████████████▉         | 7000/8920 [26:11<07:27,  4.29batch/s]--- step 7000: val_loss: 1.1468 test_loss: 1.0876  ---
loss: 2.1716:  84%|███████████████████████████████████▎      | 7500/8920 [28:20<03:36,  6.55batch/s]--- step 7500: val_loss: 1.1230 test_loss: 1.0849  ---
loss: 2.0146:  90%|█████████████████████████████████████▋    | 8000/8920 [30:20<02:16,  6.75batch/s]--- step 8000: val_loss: 1.0991 test_loss: 1.0778  ---
loss: 1.9969:  95%|████████████████████████████████████████  | 8499/8920 [32:07<01:28,  4.76batch/s]--- step 8500: val_loss: 1.0613 test_loss: 1.0696  ---
loss: 2.3871: 100%|█████████████████████████████████████████▉| 8919/8920 [33:52<00:00,  4.70batch/s]--- step 8919: val_loss: 1.0739 test_loss: 1.0487  ---
"""