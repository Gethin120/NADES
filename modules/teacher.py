import torch
import torch.nn.functional as F
from models.fe import FeatEnhancement
from models.gnn import MixGNN
from models.fd import FraudDetection
from modules.base import TrainableModule
from torch import Tensor
from torch_geometric.data import Data
from torch.nn import Linear
import torch.nn as nn


class TeacherModule(TrainableModule):
    """
    Teacher model for NADES.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.feature_enhancement = FeatEnhancement(
            in_channels, hidden_channels, dropout
        )
        self.gnn = MixGNN(hidden_channels, num_layers, dropout)
        self.fd = FraudDetection(hidden_channels, out_channels, dropout=0.1)

    def forward(self, x, edge_index):
        # 初始特征增强
        x = self.feature_enhancement(x)
        x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn(x, edge_index)

        return self.fd(x)

    def step(
        self, data, stage: str, global_weights=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        logits = self(data.x.to(device), data.edge_index.to(device))
        if stage == "train":
            mask = data.train_mask
        elif stage == "val":
            mask = data.val_mask
        else:
            mask = data.test_mask if hasattr(data, "test_mask") else None

        if mask is None:
            logits_masked = logits
            y_masked = data.y.to(device)
        else:
            logits_masked = logits[mask]
            y_masked = data.y[mask].to(device)

        loss = None
        if stage != "test":
            label_mask = y_masked != -1
            if label_mask.any():
                logits_labeled = logits_masked[label_mask]
                y_labeled = y_masked[label_mask]
                loss = F.cross_entropy(
                    logits_labeled, y_labeled, weight=global_weights)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss, logits_labeled.detach(), y_labeled.detach()

    def predict(self, data: Data) -> Tensor:
        h = self(data.x, data.edge_index)[data.batch_nodes]
        return torch.softmax(h, dim=-1)

    def reset_parameters(self):
        self.feature_enhancement.reset_parameters()
        self.trans.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()


class TeacherFL(nn.Module):
    """
    Same as TeacherModule, but with fc for BaseHeadSplit support federated learning.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.feature_enhancement = FeatEnhancement(
            in_channels, hidden_channels, dropout
        )
        self.gnn = MixGNN(hidden_channels, num_layers, dropout)
        self.fc = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.feature_enhancement(x)
        x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn(x, edge_index)
        x = self.fc(x)
        return x

    def reset_parameters(self):
        self.feature_enhancement.reset_parameters()
        self.gnn.reset_parameters()
        self.fc.reset_parameters()
