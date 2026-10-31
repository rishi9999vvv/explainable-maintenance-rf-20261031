from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class CausalConv(nn.Conv1d):
    def forward(self, x):
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(F.pad(x, (padding, 0)))

class ResidualBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.first = CausalConv(16, 16, 3, dilation=dilation)
        self.second = CausalConv(16, 16, 3, dilation=dilation)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        return F.relu(x + self.dropout(self.second(self.dropout(F.relu(self.first(x))))))

class TemporalModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.project = nn.Conv1d(n_features, 16, 1)
        self.blocks = nn.Sequential(ResidualBlock(1), ResidualBlock(2))
        self.head = nn.Conv1d(16, 1, 1)
    def sequence_logits(self, x):
        z = self.blocks(F.relu(self.project(x.transpose(1, 2))))
        return self.head(z).squeeze(1)
    def forward(self, x):
        return self.sequence_logits(x)[:, -1]

def configure_cpu(seed=42):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
