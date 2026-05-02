import torch
import torch.nn as nn

class SimpleDNAClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleDNAClassifier, self).__init__()
        self.fc1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size*2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc3 = nn.Linear(hidden_size * 2, output_size)
        self.pred = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.global_pool(out)
        out = out.squeeze(-1)  # Remove the last dimension
        out = self.fc3(out)
        out = self.pred(out)
        return out