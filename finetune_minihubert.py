from pretrain.model import MiniHubert, LinearHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from asr.datasets import get_loader
from vocab.tokenizer import Tokenizer
from tqdm import tqdm
import numpy as np
from utils import save_ckpt
from torchaudio.models.decoder import cuda_ctc_decoder
from jiwer import wer

vocab_type = 'bpe'
tokenizer = Tokenizer(vocab_type)
vocab_size = tokenizer.vocab_size
tokens = [tokenizer.spm.id_to_piece(id) for id in range(vocab_size)]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

model_dim = 768
n_clusters = 512
features_kwargs = {
    "demo0": [80, "M", 128, "M", 256, 256, "M", 512]
}


batch_size = 64
num_workers = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
loader_kwargs = {
    "tokenizer": tokenizer,
    "batch_size": batch_size,
    "num_workers": num_workers,
    "use_pesudo_label": False
}

enc_kwargs = {
    "n_embd": model_dim,
    "n_head": 6,
    "n_layer": 12,
    "block_size": 512,
    "dropout": 0.1
}

model_cfg = {
    "model_dim": model_dim,
    "n_class": n_clusters,
    "features_cfg": features_kwargs["demo0"],
    "enc_kwargs": enc_kwargs
}

# train config
num_epochs = 10
eval_interval = 500
save_begin = 1000
learning_rate = 3e-4

# decode config
nbest = 10
beam_size = 10
blank_skip_threshold = 0.95

model_name = f"finetune_minihubert_kmeans_{n_clusters}"
model_ckpts = save_ckpt(model_name)
print(f"the model checkpoints will be saved at {model_ckpts}.")

model = MiniHubert(**model_cfg).to(device)
prev_ckpt = 'checkpoints/pretrain_minihubert_kmeans_512/2024-03-15_15:31:19/best_pretrain_minihubert_kmeans_512.pth'
model.load_state_dict(torch.load(prev_ckpt))

head = LinearHead(model_dim, vocab_size).to(device)
model.final_proj = head
model.freeze()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CTCLoss()

trainloader = get_loader("train-clean-100", **loader_kwargs)
valloader = get_loader("dev-clean", **loader_kwargs)
testloader = get_loader("test-clean", **loader_kwargs)

cuda_decoder = cuda_ctc_decoder(tokens, nbest=nbest, beam_size=beam_size, blank_skip_threshold=blank_skip_threshold)

@torch.no_grad()
def estimate():
    metrics = {}
    model.eval()
    for split, loader in [('val', valloader), ('test', testloader)]:
        losses = []
        for x, y, lx, ly in tqdm(loader, ncols=100, desc=f"Eval {split} processing", leave=False):
            x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
            y_p, ly_p, _ = model(x, lx)
            log_probs = F.log_softmax(y_p, dim=-1)
            log_probs = log_probs.transpose(0, 1) # BxTxC -> TxBxC
            loss = criterion(log_probs, y, ly_p, ly)
            losses.append(loss.item())
        metrics[split + '_loss'] = np.mean(losses)

        log_probs = log_probs.transpose(0, 1) # TxBxC -> BxTxC
        ly_p = ly_p.to(torch.int32)
        results = cuda_decoder(log_probs, ly_p)
        transcripts = [tokenizer.decode(result[0].tokens).lower() for result in results]
        utterances = tokenizer.decode(y.tolist())
        metrics[split + '_wer'] = wer(utterances, transcripts)
        print("\nutterances[0]:", utterances[0])
        print("transcipts[0]:", transcripts[0])
    model.train()
    return metrics

best_loss = 100
n_batches = len(trainloader)
print(f"num_epochs: {num_epochs}")
print(f"number of batches per epoch: {n_batches}")
print(f"train loader kwargs: {loader_kwargs}")
pbar = tqdm(total=num_epochs * n_batches, desc="model finetuning", leave=True, unit="batch")
for epoch in range(num_epochs):
    for i, (x, y, lx, ly) in enumerate(trainloader):
        iter = epoch * n_batches + i
        if iter % eval_interval == 0 or iter == num_epochs * n_batches - 1:
            metrics = estimate()
            print(f"\n--- step {iter}: {metrics} ---")
            if iter > save_begin and metrics['val_loss'] < best_loss:
                best_loss = metrics['val_loss']
                torch.save(model.state_dict(), f'{model_ckpts}/best_{model_name}.pth')
        x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
        y_p, ly_p, _ = model(x, lx)
        log_probs = F.log_softmax(y_p, dim=-1)
        log_probs = log_probs.transpose(0, 1) # BxTxC -> TxBxC
        loss = criterion(log_probs, y, ly_p, ly)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"model finetuning, loss: {loss.item():.4f}")
        pbar.update(1)  
pbar.close()
