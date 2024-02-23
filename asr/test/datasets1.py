from asr.datasets import *
from vocab.tokenizer import Tokenizer

tokenizer = Tokenizer('bpe')

train_dataset = LibriSpeechDataset("train-clean-100", tokenizer)
dev_dataset = LibriSpeechDataset("dev-clean", tokenizer)
test_dataset = LibriSpeechDataset("test-clean", tokenizer)

sample_idx = 100
print(f"len(train_dataset)={len(train_dataset)}")
print(f"len(dev_dataset)={len(dev_dataset)}")
print(f"len(test_dataset)={len(test_dataset)}")
print(f"feature.shape={train_dataset[sample_idx][0].shape}")
print(f"token.shape={train_dataset[sample_idx][1].shape}")

"""
(GPT) root@asr:~/minasr# python dev.py 
len(train_dataset)=28539
len(dev_dataset)=2703
len(test_dataset)=2620
feature.shape=torch.Size([1430, 80])
token.shape=torch.Size([85])
"""