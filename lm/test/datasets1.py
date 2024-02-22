from lm.datasets import *
from vocab.tokenizer import Tokenizer

tokenizer = Tokenizer('char')
train_dataset = LibriTextDataset("train", tokenizer)
dev_dataset = LibriTextDataset("dev", tokenizer)
test_dataset = LibriTextDataset("test", tokenizer)
sample_idx = 100
print(f"len(train_dataset)={len(train_dataset)}")
print(f"len(dev_dataset)={len(dev_dataset)}")
print(f"len(test_dataset)={len(test_dataset)}")
print(f"train_dataset.transcript[{sample_idx}]={train_dataset.transcript[sample_idx].strip()}")
print(f"train_dataset[{sample_idx}].shape={train_dataset[sample_idx].shape}")

"""
(GPT) root@asr:~/minasr# python dev.py 
len(train_dataset)=281241
len(dev_dataset)=5567
len(test_dataset)=5559
train_dataset.transcript[100]=he detested the way they had of sidling past him timidly with sidewise glances as if they expected him to gobble them up at a mouthful if they ventured to say a word that was the avonlea type of well bred little girl but this freckled witch was very different
train_dataset[100].shape=torch.Size([260])
"""