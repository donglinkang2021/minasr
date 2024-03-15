import torch
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = "/opt/data/private/linkdom/data/"
FEAT_ROOT = "/opt/data/private/linkdom/data/pretrain/fbank80"
LAB_ROOT = "/opt/data/private/linkdom/data/pretrain/lab"
sample_id_root = "/opt/data/private/linkdom/data/pretrain/sample_id/libri960.txt"

class Libri960Dataset(Dataset):
    def __init__(self):
        with open(sample_id_root, "r") as f:
            sample_id_context = f.read()
        self.data = sample_id_context.strip().split("\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data[idx]
        feature_path = f"{FEAT_ROOT}/{sample_id}.pt"
        label_path = f"{LAB_ROOT}/{sample_id}.pt"
        feature = torch.load(feature_path)
        label = torch.load(label_path)
        return feature, label


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

    return data, target, data_lengths, target_lengths


def get_libri960_loader(num_workers: int, batch_size: int, shuffle: bool = True):
    dataset = Libri960Dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader