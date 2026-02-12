import torch
from torch.utils.data import Dataset, DataLoader

class DNADataset(Dataset):
    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor):
        self.sequences = sequences
        self.labels = labels
    def __len__(self) -> int:
        return len(self.sequences)
    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]
    
# Custom collate function for DataLoader
def collate_variable_length(batch):
    sequences, labels = zip(*batch)
    
        # Find max length in this batch
    max_len = max(seq.shape[-1] for seq in sequences)
    
    # Pad each sequence to max_len
    padded = []
    for seq in sequences:
        if seq.shape[-1] < max_len:
            padding = torch.zeros(seq.shape[0], max_len - seq.shape[-1])
            seq = torch.cat([seq, padding], dim=1)
        padded.append(seq)
    
    return torch.stack(padded), torch.stack(labels)