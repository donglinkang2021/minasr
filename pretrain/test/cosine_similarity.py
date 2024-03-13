import torch
from torch.nn import functional as F

input1 = torch.randn(64, 1200, 80).unsqueeze(2) # B, T, D -> B, T, 1, D
print(input1.size())
input2 = torch.randn(512, 80) # n_class, D
output = F.cosine_similarity(input1, input2, dim = -1) # (B, T, 1, D) cos (n_class, D) -> (B, T, n_class)
print(output.size())
print(output.softmax(dim=-1).size())
print(output.softmax(dim=-1).sum(dim=-1))

"""output
(GPT) root@asr:~/minasr# python pretrain/test/cosine_similarity.py
torch.Size([64, 1200, 1, 80])
torch.Size([64, 1200, 512])
torch.Size([64, 1200, 512])
tensor([[1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
        ...,
        [1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000]])
"""