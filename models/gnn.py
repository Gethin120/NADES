import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Dropout, Module, BatchNorm1d
from torch_geometric.nn import GraphSAGE, GCN, GAT
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TransformerConv


class MixGNN(nn.Module):
    """
    融合了 Transformer 和 SAGE 的 GNN 模块
    """

    def __init__(
        self,
        hidden_channels,
        num_layers,
        dropout,
        heads=1,
        use_residual=True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.use_residual = use_residual
        self.dropout = dropout
        self.residual_alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的残差连接权重
        self.trans = TransformerConv(
            hidden_channels,
            hidden_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
        )  # GNN 层内置的 dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index):
        x_trans = self.trans(x, edge_index)
        x = F.relu(x_trans)
        x = F.dropout(x, p=self.dropout, training=self.training)
        prev_x = x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if not (self.training and x.size(0) <= 1):
                x = self.batch_norms[i](x)
            if self.use_residual and prev_x.shape == x.shape:
                alpha = torch.sigmoid(self.residual_alpha)  # 计算残差连接权重
                x = alpha * x + (1 - alpha) * prev_x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            prev_x = x

        return x


class GNN(Module):
    """
    A flexible GNN model.
    This implementation supports plain_last option.
    """

    def __init__(
        self,
        *,
        conv: str,
        output_dim: int,
        num_layers: int,
        hidden_dim: int = 16,
        dropout: float = 0.0,
        batch_norm: bool = False,
        jk: str = None,
        plain_last: bool = True,
        **conv_kwargs,
    ):
        super().__init__()

        model_kwargs = dict(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm="batchnorm" if batch_norm else None,
            jk=jk,
        )

        if conv == "sage":
            self.model = GraphSAGE(
                project=False,
                **model_kwargs,
                **conv_kwargs,
            )
        elif conv == "gcn":
            self.model = GCN(
                **model_kwargs,
                **conv_kwargs,
            )
        elif conv == "gat":
            self.model = GAT(
                **model_kwargs,
                **conv_kwargs,
            )
        else:
            raise NotImplementedError(f"Unknown conv type: {conv}")

        self.dropout_fn = Dropout(dropout, inplace=True)
        self.batch_norm = batch_norm
        self.plain_last = plain_last
        if not plain_last and batch_norm:
            self.bn = BatchNorm1d(output_dim)

    def forward(self, x: Tensor, adj_t: Tensor) -> Tensor:
        x = self.model(x, adj_t)
        if not self.plain_last:
            if self.batch_norm and not (self.training and x.size(0) <= 1):
                x = self.bn(x)
            x = self.dropout_fn(x)
            x = F.relu(x)
        return x

    def reset_parameters(self):
        self.model.reset_parameters()
        if hasattr(self, "bn"):
            self.bn.reset_parameters()
