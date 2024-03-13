from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm
from pathlib import Path
import torch
import joblib

SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]

DATA_ROOT = "/opt/data/private/linkdom/data/"
out_root = "/opt/data/private/linkdom/data/pretrain/"

device = "cuda" if torch.cuda.is_available() else "cpu"
n_clusters = 512
km_model = joblib.load(f"{out_root}/kmeans_{n_clusters}")
codebooks = km_model.cluster_centers_
codebooks = torch.from_numpy(codebooks).to(device)
print(f"Kmeans model loaded from {out_root}, codebooks shape: {codebooks.shape}")

for split in SPLITS:
    out_root = Path(out_root).absolute()
    feature_root = out_root / "fbank80" / split 
    labels_root = out_root / "lab" / split
    labels_root.mkdir(parents = True, exist_ok = True)
    print(f"the kmeans label will be save at {labels_root}")
    dataset = LIBRISPEECH(root = DATA_ROOT, url = split, download = False)
    print("Dumping kmeans label...")
    pbar = tqdm(total = len(dataset), desc = "Kmeans label", leave = True)
    for waveform, sample_rate, utterance, spk_id, chapter_no, utt_no in dataset:
        sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
        feature = torch.load(feature_root / f"{sample_id}.pt")
        pesudo_label = torch.cdist(feature, codebooks).argmin(dim=-1)
        pesudo_label_path = labels_root / f"{sample_id}.pt"
        torch.save(pesudo_label, pesudo_label_path)
        pbar.set_description(f"Dumping kmeans label for {split} for {sample_id}")
        pbar.update(1)
    pbar.close()
