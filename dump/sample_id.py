from torchaudio.datasets import LIBRISPEECH
from asr.kaldi import FBANK80
from tqdm import tqdm
from pathlib import Path
import torch

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
out_root = Path(out_root).absolute()
sample_id_root = out_root / "sample_id"  
sample_id_root.mkdir(exist_ok=True)

for split in SPLITS:
    print(f"sample id will be save at {sample_id_root}")
    sample_id_context = ""
    dataset = LIBRISPEECH(root = DATA_ROOT, url = split, download = False)
    pbar = tqdm(total = len(dataset), desc = "Feature extraction", leave = True)
    for _, _, _, spk_id, chapter_no, utt_no in dataset:
        sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
        sample_id_context += sample_id + "\n"
        pbar.set_description(f"Processing {sample_id}")
        pbar.update(1)
    pbar.close()
    
    sample_id_path = sample_id_root / f"{split}.txt"
    with open(sample_id_path, "w") as f:
        f.write(sample_id_context)
    print(f"sample id for {split} has been saved at {sample_id_path}")
