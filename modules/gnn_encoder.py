import torch.nn as nn
import torch.nn.functional as F
from models.fe import FeatEnhancement
from models.gnn import MixGNN


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.feature_enhancement = FeatEnhancement(
            in_channels, hidden_channels, dropout
        )
        self.gnn = MixGNN(hidden_channels, num_layers, dropout)

    def forward(self, x, edge_index):
        x = self.feature_enhancement(x)
        x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn(x, edge_index)


        return x 
