import joblib
import torch

KM_ROOT = "/opt/data/private/linkdom/data/pretrain/"

def load_codebooks(n_clusters: int):
    km_model = joblib.load(f"{KM_ROOT}/kmeans_{n_clusters}")
    codebooks = km_model.cluster_centers_
    codebooks = torch.from_numpy(codebooks)
    return codebooks

n_clusters = 256
codebooks = load_codebooks(n_clusters)
print(f"codebooks shape: {codebooks.shape}")
torch.save(codebooks, f"{KM_ROOT}/codebooks_{n_clusters}x80.pt")