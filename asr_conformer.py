import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from asr.datasets import get_loader
from asr.model import ConformerCTC
from vocab.tokenizer import Tokenizer
import numpy as np
from utils import save_ckpt
from torchaudio.models.decoder import cuda_ctc_decoder
from jiwer import wer

# tokenizer config
vocab_type = 'bpe'
tokenizer = Tokenizer(vocab_type)
vocab_size = tokenizer.vocab_size
tokens = [tokenizer.spm.id_to_piece(id) for id in range(vocab_size)]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# loader config
batch_size = 64
num_workers = 4
loader_kwargs = {
    "tokenizer": tokenizer,
    "batch_size": batch_size,
    "num_workers": num_workers,
    "use_pesudo_label": False
}

# model config
model_dim = 768

features_kwargs = {
    "demo0": [80, "M", 128, "M", 512],
    "demo1": [80, "M", 128, "M", 512, "M", 768],
    "demo2": [80, "M", 128, "M", 512, "M", 768, "M", 1024],
}

conformer_kwargs = {
    "input_dim": model_dim,
    "num_heads": 6,
    "ffn_dim": 128,
    "num_layers": 4,
    "depthwise_conv_kernel_size": 31,
    "dropout": 0.2,
}

modelctc_kwargs = {
    "features_cfg": features_kwargs["demo0"],
    "conformer_kwargs": conformer_kwargs,
    "vocab_size": vocab_size
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

# ------------------- asr train and eval -------------------
model = ConformerCTC(**modelctc_kwargs).to(device)
print(f"number of parameters of asr model: {model.get_num_params()/1e6:.6f} M ")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
model_name = f"asr_conformer"
model_ckpts = save_ckpt(model_name)
print(f"the model checkpoints will be saved at {model_ckpts}.")

cuda_decoder = cuda_ctc_decoder(tokens, nbest=nbest, beam_size=beam_size, blank_skip_threshold=blank_skip_threshold)

trainloader = get_loader("train-clean-100", **loader_kwargs)
valloader = get_loader("dev-clean", **loader_kwargs)
testloader = get_loader("test-clean", **loader_kwargs)

@torch.no_grad()
def estimate():
    metrics = {}
    model.eval()
    for split, loader in [('val', valloader), ('test', testloader)]:
        losses = []
        for x, y, lx, ly in tqdm(loader, ncols=100, desc=f"Eval {split} processing", leave=False):
            x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
            logits, l_logits = model(x, lx)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1) # BxTxC -> TxBxC
            loss = criterion(log_probs, y, l_logits, ly)
            losses.append(loss.item())
        metrics[split + '_loss'] = np.mean(losses)

        log_probs = log_probs.transpose(0, 1) # TxBxC -> BxTxC
        l_logits = l_logits.to(torch.int32)
        results = cuda_decoder(log_probs, l_logits)
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
pbar = tqdm(total=num_epochs * n_batches, desc="model training", leave=True, unit="batch")
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
        logits, l_logits = model(x, lx)
        log_probs = logits.log_softmax(dim=-1)
        log_probs = log_probs.transpose(0, 1) # BxTxC -> TxBxC
        loss = criterion(log_probs, y, l_logits, ly)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"model training, loss: {loss.item():.4f}")
        pbar.update(1)  
pbar.close()