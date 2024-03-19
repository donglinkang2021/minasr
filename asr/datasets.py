import torch
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Dataset, DataLoader

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
FEAT_ROOT = "/opt/data/private/linkdom/data/pretrain/fbank80"
LAB_ROOT = "/opt/data/private/linkdom/data/pretrain/lab"
sample_id_root = "/opt/data/private/linkdom/data/pretrain/sample_id"

class LibriSpeechDataset(Dataset):
    def __init__(self, split: str, tokenizer, use_pesudo_label: bool):
        self.split = split
        if use_pesudo_label == False:
            self.data = LIBRISPEECH(root = DATA_ROOT, url = split, download = False)
        else:
            sample_id_path = f"{sample_id_root}/{split}.txt"
            with open(sample_id_path, "r") as f:
                sample_id_context = f.read()
            self.data = sample_id_context.strip().split("\n")
        self.use_pesudo_label = use_pesudo_label
        self.feature_root = f"{FEAT_ROOT}/{split}"
        self.label_root = f"{LAB_ROOT}/{split}"
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, _, utterance, spk_id, chapter_no, utt_no = self.data[idx]
        sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
        feature_path = f"{self.feature_root}/{sample_id}.pt"
        feature = torch.load(feature_path)
        label = torch.LongTensor(self.tokenizer.encode(utterance.lower()))
        
        # sample_id = self.data[idx]
        # feature_path = f"{self.feature_root}/{sample_id}.pt"
        # feature = torch.load(feature_path)
        # label_path = f"{self.label_root}/{sample_id}.pt"
        # label = torch.load(label_path)
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

    # unpad labels and concatenate them for CTC loss
    # target = [item[1] for item in sorted_batch]
    # target_lengths = torch.tensor([t.size(0) for t in target],dtype=torch.long)
    # target = torch.cat(target, dim=0)

    return data, target, data_lengths, target_lengths


def get_loader(split: str, tokenizer, num_workers: int, batch_size: int, shuffle: bool = True, use_pesudo_label: bool = False):
    dataset = LibriSpeechDataset(split, tokenizer, use_pesudo_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader