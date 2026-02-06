import torch.nn as nn
import torch.nn.functional as F


class FeatEnhancement(nn.Module):
    """
    Feature Enhancement Module for the input features. The output features are used for the GNN encoder.
    """

    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = out_channels
        self.dropout = dropout
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
