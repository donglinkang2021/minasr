# generate the transcript file of all the LibriSpeech data
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm
from pathlib import Path

SPLITS = [
    # "dev-clean",  
    # "dev-other", 
    # "test-clean", 
    # "test-other", 
    "train-clean-100", 
    "train-clean-360", 
    "train-other-500",
]

LS_ROOT = "/opt/data/private/linkdom/data/"
out_root = "/opt/data/private/linkdom/data/libritext"
out_root = Path(out_root).absolute()
transcript_file = out_root / "transcript.txt"

# open the transcript file and write the utterances
with transcript_file.open("w") as f:
    for split in SPLITS:
        dataset = LIBRISPEECH(root = LS_ROOT, url = split, download = False)
        print(f"Processing {split}...")
        for _, _, utter, _, _, _ in tqdm(dataset):
            f.write(f"{utter.lower()}\n")

"""
Processing train-clean-100...
100%|██████████| 28539/28539 [12:55<00:00, 36.81it/s]
Processing train-clean-360...
100%|██████████| 104014/104014 [22:02<00:00, 78.64it/s] 
Processing train-other-500...
100%|██████████| 148688/148688 [1:03:33<00:00, 38.99it/s]
"""