from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm
from pathlib import Path
import torch
import joblib

SPLITS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
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


"""
(GPT) root@asr:~/minasr# python dump/pesudo_labels.py 
Kmeans model loaded from /opt/data/private/linkdom/data/pretrain/, codebooks shape: torch.Size([512, 80])
Dumping kmeans label for dev-clean for 8842-304647-13: 100%|█████████| 2703/2703 [02:16<00:00, 19.78it/s]
Dumping kmeans label for dev-other for 8288-274162-66: 100%|█████████| 2864/2864 [05:58<00:00,  8.00it/s]
Dumping kmeans label for test-clean for 908-31957-25: 100%|██████████| 2620/2620 [03:54<00:00, 11.16it/s]
Dumping kmeans label for test-other for 8461-281231-38: 100%|██████████| 2939/2939 [04:49<00:00, 10.16it/s]
Dumping kmeans label for train-clean-100 for 911-130578-20: 100%|██████████| 28539/28539 [22:16<00:00, 21.35it/s]
Dumping kmeans label for train-clean-360 for 986-129388-112: 100%|██████████| 104014/104014 [1:54:46<00:00, 15.10it/s]
Dumping kmeans label for train-other-500 for 985-126228-51: 100%|██████████| 148688/148688 [3:07:26<00:00, 13.22it/s]
"""