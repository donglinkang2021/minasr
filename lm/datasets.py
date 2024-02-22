import torch
from torch.utils.data import Dataset, DataLoader

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