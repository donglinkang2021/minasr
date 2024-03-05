import torch
import torch.nn as nn
from torch.nn import functional as F


utt_vocab_size = 1000
model_dim = 512
logits = torch.randn(20, utt_vocab_size).requires_grad_()
target = torch.randint(0, utt_vocab_size, (20,))
# Sample soft categorical using reparametrization trick
idx_onehot = F.gumbel_softmax(logits, tau=1, hard=True)
embed = nn.Embedding(utt_vocab_size, model_dim)
print("idx_onehot:", idx_onehot.shape)
print("embed.weight:", embed.weight.shape)
embeddings = idx_onehot @ embed.weight
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)
loss.backward()
print("logits:", logits)
print("loss:", loss)