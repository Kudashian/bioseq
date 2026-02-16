import torch.nn as nn
import torch

class PromoterCNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=10, padding=5)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=10, padding=5)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.pred = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.global_pool(out)
        out = out.squeeze(-1)  # Remove the last dimension
        out = self.dropout(out)
        out = self.fc(out)
        out = self.pred(out)
        return out