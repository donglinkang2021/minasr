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
    feature_root.mkdir(parents = True, exist_ok = True)
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
        pbar.set_description(f"Processing {sample_id} waveform.shape={waveform.shape} feature.shape={feature.shape}")
        pbar.update(1)
    pbar.close()


"""
(GPT) root@asr:~/minasr# python pesudo_labels.py 
Kmeans model loaded from /opt/data/private/linkdom/data/pretrain/, codebooks shape: torch.Size([512, 80])
the kmeans label will be save at /opt/data/private/linkdom/data/pretrain/lab/train-clean-100
Dumping kmeans label...
pesudo_label=tensor([ 53,  13, 276,  ...,  91,  91,  91])
pesudo_label.shape=torch.Size([1407])
the kmeans label will be save at /opt/data/private/linkdom/data/pretrain/lab/train-clean-360
Dumping kmeans label...
pesudo_label=tensor([237, 477, 388, 376, 364, 492, 221, 486, 228, 228, 228, 228, 228, 228,
        228,  80,  80, 296, 228, 427, 282, 492, 235, 473, 309, 235, 228, 228,
        348, 466, 485, 485, 385, 221, 385, 485, 264, 131, 131, 131, 131, 420,
        429, 429,  60,  60,  60,  60,  60,  60,  60,  60,  60,  60,  60,  60,
         60,  60,  60,  60,  60, 192,  51,  51,  51,  51,  51,  51,  51,  51,
         51, 429, 429, 429, 429, 278,  51,  51, 429, 171, 309, 171,  18, 505,
        131, 131, 395,  60,  44, 120, 120, 120, 120, 120, 336, 336, 287, 175,
        175, 171, 171, 228, 171, 385, 283, 283, 143, 143, 268, 268, 268, 268,
        268, 268, 268, 336, 336, 336,   1, 292, 107, 178, 485, 178, 178, 136,
        136, 136, 485, 136, 136, 136, 427, 136, 316, 316, 136, 278, 248, 248,
        248, 106, 106, 106,  14,  14,  14,  14,  14,  14,  87,  87,  87,  87,
         87, 265, 265, 265, 265, 132, 266, 266, 266, 266, 266, 266, 266, 266,
        132,   3, 373,  75, 250,  52, 131, 131, 298, 298, 207,  80, 228, 423,
        309, 434, 309, 423, 365, 365, 412, 473, 432,  99, 309, 228, 309, 412,
        423, 309, 309, 228, 423, 434])
pesudo_label.shape=torch.Size([202])
the kmeans label will be save at /opt/data/private/linkdom/data/pretrain/lab/train-other-500
Dumping kmeans label...
pesudo_label=tensor([352, 231, 231,  ..., 392, 392, 232])
pesudo_label.shape=torch.Size([1517])
"""