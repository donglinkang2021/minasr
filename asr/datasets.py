import torch
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Dataset, DataLoader
from asr.kaldi import FBANK80

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

class LibriSpeechDataset(Dataset):
    def __init__(self, split: str, tokenizer):
        self.split = split
        self.data = LIBRISPEECH(root = DATA_ROOT, url = split, download = False)
        self.tokenizer = tokenizer
        self.transform = FBANK80()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform, _, utterance, _, _, _ = self.data[idx]
        feature = self.transform(waveform)
        token = torch.LongTensor(self.tokenizer.encode(utterance.lower()))
        return feature, token


def collate_fn(batch):
    # Sorting sequences by lengths
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)

    # Pad data sequences
    data = [item[0].squeeze() for item in sorted_batch]
    data_lengths = torch.tensor([len(d) for d in data],dtype=torch.long) 
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)

    # Pad labels
    target = [item[1] for item in sorted_batch]
    target_lengths = torch.tensor([t.size(0) for t in target],dtype=torch.long)
    target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=0)

    # unpad labels and concatenate them for CTC loss
    # target = [item[1] for item in sorted_batch]
    # target_lengths = torch.tensor([t.size(0) for t in target],dtype=torch.long)
    # target = torch.cat(target, dim=0)

    return data, target, data_lengths, target_lengths


def get_loader(split: str, tokenizer, num_workers, batch_size: int, shuffle: bool = True):
    dataset = LibriSpeechDataset(split, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader