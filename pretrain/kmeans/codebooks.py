import torch

KM_ROOT = "/opt/data/private/linkdom/data/pretrain/"

def load_codebooks(n_clusters: int) -> torch.Tensor:
    codebooks = torch.load(f"{KM_ROOT}/codebooks_{n_clusters}x80.pt")
    return codebooks

# n_clusters = 256
# codebooks = load_codebooks(n_clusters)
# print(f"codebooks shape: {codebooks.shape}")

"""
(GPT) root@asr:~/minasr# python pretrain/kmeans/codebooks.py
codebooks shape: torch.Size([256, 80])
"""