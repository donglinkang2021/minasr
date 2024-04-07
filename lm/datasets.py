import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L

DATA_ROOT = "/opt/data/private/linkdom/data/libritext/"

class LibriTextDataset(Dataset):
    def __init__(self, split: str, tokenizer, max_len: int = 512):
        assert split in ["train", "dev", "test"]
        self.split = split
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transcript_file = f"{DATA_ROOT}/{split}/transcript.txt"
        self.transcript = self._load_transcript()

    def _load_transcript(self):
        with open(self.transcript_file, "r") as f:
            transcript = f.readlines()
        return transcript
    
    def __len__(self):
        return len(self.transcript)
    
    def __getitem__(self, idx):
        utterance = self.transcript[idx].strip()
        tokens = self.tokenizer.encode(utterance)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch):
    paded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    x = paded_batch[:, :-1]
    y = paded_batch[:, 1:]
    return x, y

def get_loader(split: str, tokenizer, batch_size: int, num_workers: int, max_len: int = 512):
    dataset = LibriTextDataset(split, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    return loader


class LibriTextDataModule(L.LightningDataModule):
    def __init__(self, tokenizer, train_set: str, dev_set: str, test_set: str,
                 batch_size: int, num_workers: int, max_len: int = 512):
        super().__init__()
        self.load_kwargs = {
            "tokenizer": tokenizer,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "max_len": max_len
        }
        self.dataset_names = {
            "train": train_set,
            "dev": dev_set,
            "test": test_set
        }

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_loader = get_loader(self.dataset_names["train"], **self.load_kwargs)
            self.val_loader = get_loader(self.dataset_names["dev"], **self.load_kwargs)
        if stage == 'test' or stage is None:
            self.test_loader = get_loader(self.dataset_names["test"], **self.load_kwargs)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader