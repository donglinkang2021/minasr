import torch
import joblib
from asr.datasets import get_loader
from vocab.tokenizer import Tokenizer

n_clusters = 512 # 256 or 512
vocab_type = 'bpe'
tokenizer = Tokenizer(vocab_type)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
loader_kwargs = {
    "tokenizer": tokenizer,
    "batch_size": 64,
    "num_workers": 4
}
trainloader = get_loader("train-clean-360", **loader_kwargs)

@torch.no_grad()
def kmeans_label(x: torch.Tensor, codebooks: torch.Tensor) -> torch.LongTensor:
    return torch.cdist(x, codebooks).argmin(dim=-1)

km_path = "/opt/data/private/linkdom/data/pretrain/"
km_model = joblib.load(f"{km_path}/kmeans_{n_clusters}")
codebooks = km_model.cluster_centers_
codebooks = torch.from_numpy(codebooks).to(device)
print(f"Kmeans model loaded from {km_path}, codebooks shape: {codebooks.shape}")

best_loss = 100
n_batches = len(trainloader)
print(f"number of batches per epoch: {n_batches}")
print(f"train loader kwargs: {loader_kwargs}")
for i, (x, _, lx, _) in enumerate(trainloader):
    x, lx = x.to(device), lx.to(device)
    pesudo_label = kmeans_label(x, codebooks)
    print(f"batch {i+1}/{n_batches}, pesudo_label shape: {pesudo_label.shape}")
    print(f"pesudo_label[0][:100]: {pesudo_label[0][:100]}")
    break

"""
(GPT) root@asr:~/minasr# python kmeans_test.py 
Kmeans model loaded from /opt/data/private/linkdom/data/pretrain/, codebooks shape: torch.Size([256, 80])
number of batches per epoch: 1626
train loader kwargs: {'tokenizer': <Tokenizer vocab_size=1000 token_type=bpe>, 'batch_size': 64, 'num_workers': 4}
batch 1/1626, pesudo_label shape: torch.Size([64, 1636])
pesudo_label[0][:100]: tensor([ 17,  50, 145, 145, 244, 145, 244, 244, 244,  17,  17,  17, 244, 244,
        244, 244, 244, 244, 244,  83,  83,  83,  83,  83, 131, 201, 201, 201,
         86,  35, 202, 141, 166, 166, 166, 166, 166, 185, 185, 185, 185,  12,
         12,  96,  12,  12,  12,  12, 185, 185, 166, 166, 166, 166, 166, 166,
        166, 166, 214, 214, 214, 143, 143, 143,  31,  54,  91,  91,  91,  91,
         91,  91,  91,  91,  91,  91,  91, 202, 166, 110, 142, 255, 255, 255,
        255, 188, 188, 188, 188, 188, 188, 202,  54,  54,  91, 205,  54,  91,
         91,  91])
(GPT) root@asr:~/minasr# python kmeans_test.py 
Kmeans model loaded from /opt/data/private/linkdom/data/pretrain/, codebooks shape: torch.Size([512, 80])
number of batches per epoch: 1626
train loader kwargs: {'tokenizer': <Tokenizer vocab_size=1000 token_type=bpe>, 'batch_size': 64, 'num_workers': 4}
batch 1/1626, pesudo_label shape: torch.Size([64, 1636])
pesudo_label[0][:100]: tensor([221, 440, 440, 298, 298, 298, 298, 221, 221, 221, 221, 221, 221, 298,
        298, 298, 207, 262, 262,  86,  86, 131, 131, 131, 273, 228, 228,  80,
        282, 439,  83,  30, 227, 227, 227, 288, 108, 288, 288, 288,   4,   4,
         60, 494, 494, 494, 494, 288, 288, 288, 108, 108, 108, 108, 108, 108,
         51,  51,  51,  51,  51,  51, 347, 347, 242, 464, 316, 340, 340, 340,
        340, 340, 340, 340, 340, 136, 316, 155, 141, 141, 185, 141, 141, 141,
        172, 172, 172, 172, 405, 405,  83, 464, 372, 372, 340, 372, 372, 340,
        372, 340])
pesudo_label[0][-100:]: tensor([127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
        127, 127])
"""