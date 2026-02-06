import torch.nn as nn
import torch.nn.functional as F

# from torch_geometric.nn import SAGEConv, TransformerConv
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
        # self.convs = nn.ModuleList()
        # self.batch_norms = nn.ModuleList()

        # [建议修改]：将 feature_enhancement 和 trans 也视为编码器的一部分
        # self.feature_enhancement = nn.Sequential(
        #     nn.Linear(in_channels, hidden_channels),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_channels, hidden_channels),
        # )
        # self.trans = TransformerConv(hidden_channels, hidden_channels, heads=1)

        # for _ in range(self.num_layers):
        #     self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        #     self.batch_norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, x, edge_index):
        x = self.feature_enhancement(x)
        x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn(x, edge_index)
        # x = self.trans(x, edge_index)

        # for i in range(self.num_layers):
        #     x = self.convs[i](x, edge_index)
        #     x = self.batch_norms[i](x)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)

        return x  # 最终的节点表示 h_v
