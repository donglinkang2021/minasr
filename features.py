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
    # "train-clean-100",
    "train-clean-360",
    "train-other-500",
]

DATA_ROOT = "/opt/data/private/linkdom/data/"
out_root = "/opt/data/private/linkdom/data/pretrain/"

# split = SPLITS[5]

for split in SPLITS:
    out_root = Path(out_root).absolute()
    out_root.mkdir(exist_ok=True)
    feature_root = out_root / "fbank80" / split 
    feature_root.mkdir(exist_ok=True)
    print(f"the features will be save at {feature_root}")
    dataset = LIBRISPEECH(root = DATA_ROOT, url = split, download = False)
    transform = FBANK80()
    print("Extracting log mel filter bank features...")
    pbar = tqdm(total = len(dataset), desc = "Feature extraction", leave = True)
    for waveform, sample_rate, utterance, spk_id, chapter_no, utt_no in dataset:
        sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
        feature = transform(waveform)
        feature_path = feature_root / f"{sample_id}.pt"
        torch.save(feature, feature_path)
        pbar.set_description(f"Processing {sample_id} waveform.shape={waveform.shape} feature.shape={feature.shape}")
        pbar.update(1)
    pbar.close()

"""output
(GPT) root@asr:~/minasr# python features.py 
the features will be save at /opt/data/private/linkdom/data/pretrain/fbank80/train-clean-100
Extracting log mel filter bank features...
Processing 911-130578-20 waveform.shape=torch.Size([1, 134640]) feature.shape=torch.Size([840, 80]): 100%|█████████████████████████| 28539/28539 [19:14<00:00, 24.71it/s]
"""