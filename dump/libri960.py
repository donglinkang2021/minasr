from tqdm import tqdm
from pathlib import Path

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

"""
100%|██████████| 28539/28539 [00:00<00:00, 2057889.21it/s]
sample id for train-clean-100 has been read
100%|██████████| 104014/104014 [00:00<00:00, 1574678.80it/s]
sample id for train-clean-360 has been read
100%|██████████| 148688/148688 [00:00<00:00, 2281121.58it/s]
sample id for train-other-500 has been read
sample id for whole dataset has been saved at /opt/data/private/linkdom/data/pretrain/sample_id/libri960.txt
"""