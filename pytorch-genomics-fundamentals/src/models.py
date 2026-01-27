import torch
import torch.nn as nn

class SimpleDNAClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleDNAClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.pred = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.pred(out)
        return out