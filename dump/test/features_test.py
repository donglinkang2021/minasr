from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm
from pathlib import Path
import torch

SPLITS = [
    # "dev-clean",
    # "dev-other",
    # "test-clean",
    # "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]

DATA_ROOT = "/opt/data/private/linkdom/data/"
out_root = "/opt/data/private/linkdom/data/pretrain/"

for split in SPLITS:
    out_root = Path(out_root).absolute()
    feature_root = out_root / "fbank80" / split 
    dataset = LIBRISPEECH(root = DATA_ROOT, url = split, download = False)
    print("Testing log mel filter bank features...")
    pbar = tqdm(total = len(dataset), desc = "Feature extraction", leave = True, mininterval=2)
    for waveform, sample_rate, utterance, spk_id, chapter_no, utt_no in dataset:
        sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
        feature_path = feature_root / f"{sample_id}.pt"
        # feature = torch.load(feature_path)
        # check if the feature file exists
        assert feature_path.exists(), f"Feature file {feature_path} does not exist!"
        pbar.set_description(f"Testing {sample_id}")
        pbar.update(1)
    pbar.close()
