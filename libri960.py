from torchaudio.datasets import LIBRISPEECH
from asr.kaldi import FBANK80
from tqdm import tqdm
from pathlib import Path
import torch

SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]

DATA_ROOT = "/opt/data/private/linkdom/data/"
out_root = "/opt/data/private/linkdom/data/pretrain/"
out_root = Path(out_root).absolute()
sample_id_root = out_root / "sample_id"  

sample_id_context = ""

for split in SPLITS:
    with open(sample_id_root / f"{split}.txt", "r") as f:
        data = f.readlines()
        for line in tqdm(data):
            sample_id_context += f"{split}/{line}"
        print(f"sample id for {split} has been read")

sample_id_path = sample_id_root / f"libri960.txt"
with open(sample_id_path, "w") as f:
    f.write(sample_id_context)
print(f"sample id for whole dataset has been saved at {sample_id_path}")
