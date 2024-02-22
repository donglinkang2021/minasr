from lm.datasets import *
from vocab.tokenizer import Tokenizer
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer('char')
train_kwargs = {
    "split": "train", # "train", "dev", "test
    "tokenizer": tokenizer,
    "batch_size": 64,
    "num_workers": 2
}
trainloader = get_loader(**train_kwargs)

n_batches = len(trainloader)
print(f"number of batches per epoch: {n_batches}")
print(f"train loader kwargs: {train_kwargs}")

# create a progress bar
pbar = tqdm(total=n_batches, desc="Processing batches", ncols=200,  leave=True, unit="batch")
for x, y in trainloader:
    x, y = x.to(device), y.to(device)
    pbar.set_description(f"x.shape: {x.shape} y.shape: {y.shape}")
    pbar.update(1)  
pbar.close()

"""different num_workers it seems that 0 is the fastest
(GPT) root@asr:~/minasr# python dev.py 
number of batches per epoch: 8789
train loader kwargs: {'split': 'train', 'tokenizer': <Tokenizer vocab_size=30 token_type=char>, 'batch_size': 32, 'num_workers': 4}
x.shape: torch.Size([25, 328]) y.shape: torch.Size([25, 328]): 100%|████████████████████████████████████████████████████████████████████████████████████████████| 8789/8789 [01:12<00:00, 121.01batch/s]
(GPT) root@asr:~/minasr# 
(GPT) root@asr:~/minasr# python dev.py 
number of batches per epoch: 8789
train loader kwargs: {'split': 'train', 'tokenizer': <Tokenizer vocab_size=30 token_type=char>, 'batch_size': 32, 'num_workers': 2}
x.shape: torch.Size([25, 281]) y.shape: torch.Size([25, 281]): 100%|████████████████████████████████████████████████████████████████████████████████████████████| 8789/8789 [00:55<00:00, 157.77batch/s]
(GPT) root@asr:~/minasr# 
(GPT) root@asr:~/minasr# python dev.py 
number of batches per epoch: 8789
train loader kwargs: {'split': 'train', 'tokenizer': <Tokenizer vocab_size=30 token_type=char>, 'batch_size': 32, 'num_workers': 0}
x.shape: torch.Size([25, 273]) y.shape: torch.Size([25, 273]): 100%|████████████████████████████████████████████████████████████████████████████████████████████| 8789/8789 [00:43<00:00, 201.91batch/s]
"""