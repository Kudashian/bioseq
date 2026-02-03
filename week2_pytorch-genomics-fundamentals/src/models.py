import torch
import torch.nn as nn

class SimpleDNAClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seq_length: int):
        super(SimpleDNAClassifier, self).__init__()
        self.fc1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size * seq_length, output_size)
        self.pred = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc2(out)
        out = self.pred(out)
        return out