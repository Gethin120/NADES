import torch.nn as nn
from torch_geometric.utils import dropout_adj
from models.fd import FraudDetection
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # 您可以使用您自己的 GNN 层
from torch_geometric.nn.models import DeepGraphInfomax


class SSLModel(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder  # 传入 GNNEncoder 的实例

    def forward(self, x, edge_index):
        raise NotImplementedError("Subclasses must implement forward method")

    def test(self, x, edge_index):
        self.encoder.eval()


class Grace(SSLModel):

    def __init__(
        self,
        encoder,
        hidden_channels: int,
        projection_dim: int,
    ):
        super().__init__(encoder)
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x, edge_index):
        # 1. 编码
        h_v = self.encoder(x, edge_index)
        # 2. 投影
        return self.projector(h_v)


class Infomax(SSLModel):

    def __init__(self, encoder, hidden_channels: int):
        super().__init__(encoder)
        # 包装 DeepGraphInfomax
        self.dgi_model = DeepGraphInfomax(
            hidden_channels=hidden_channels,
            encoder=encoder,
            summary=self._summary,
            corruption=self._corruption,
        )

    def _corruption_edge_drop(self, x, edge_index, p_drop=0.2):

        # dropout_adj 返回 (剩余的 edge_index, 剩余的 edge_mask)
        # 我们设置 force_undirected=True 来确保对称性
        edge_index_dropped, _ = dropout_adj(
            edge_index,
            p=p_drop,
            force_undirected=True,
            training=True,  # 确保 dropout 生效
        )

        # 负样本 = 原始特征 + 残缺的结构
        return x, edge_index_dropped

    def _corruption_feature_mask(self, x, edge_index, p_mask=0.1):

        num_nodes = x.size(0)

        # 生成要遮盖的节点索引
        num_nodes_to_mask = int(num_nodes * p_mask)
        nodes_to_mask = torch.randperm(num_nodes)[:num_nodes_to_mask]

        # 复制特征并置零
        x_corrupted = x.clone()
        x_corrupted[nodes_to_mask] = 0.0

        # 负样本 = 残缺的特征 + 原始结构
        return x_corrupted, edge_index

    def _corruption(self, x, edge_index):
        perm = torch.randperm(x.size(0), device=x.device)
        return x[perm], edge_index

    def _summary(self, z, *args, **kwargs):
        return torch.sigmoid(z.mean(dim=0))

    def forward(self, x, edge_index):

        return self.dgi_model(x, edge_index)

    def loss(self, pos_z, neg_z, summary_vec):

        return self.dgi_model.loss(pos_z, neg_z, summary_vec)


class GraphMAE(SSLModel):


    def __init__(
        self,
        encoder: nn.Module,
        in_channels: int,  # 原始特征维度 (e.g., 128)
        out_channels: int,  # 编码器输出维度 (e.g., 64)
        mask_rate: float = 0.5,
        decoder_dim: int = 64,  # 解码器中间维度
        replace_with_zero: bool = False,  # 是用 0 遮盖还是用可学习 Mask Token
    ):
        super().__init__(encoder)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_rate = mask_rate
        self.replace_with_zero = replace_with_zero

        # 1. 可学习的 Mask Token
        if not self.replace_with_zero:
            self.mask_token = nn.Parameter(torch.randn(1, in_channels))

        # 2. 解码器 (Decoder)
        # 一个简单的 1 层 GNN 解码器 + 线性层
        self.decoder_gnn = GCNConv(out_channels, decoder_dim)
        self.decoder_mlp = nn.Linear(decoder_dim, in_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):


        # --- 1. Masking ---
        num_nodes = x.size(0)
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(num_nodes * self.mask_rate)
        mask_indices = perm[:num_mask_nodes]

        x_masked = x.clone()
        if self.replace_with_zero:
            x_masked[mask_indices] = 0.0
        else:
            x_masked[mask_indices] = self.mask_token

        # --- 2. Encoding ---
        # 编码器在“损坏”的图上工作
        z = self.encoder(x_masked, edge_index)

        # --- 3. Decoding ---
        # 解码器重建特征
        h_decode = F.relu(self.decoder_gnn(z, edge_index))
        x_reconstructed = self.decoder_mlp(h_decode)

        # 返回 重建结果, 原始目标, 遮盖索引
        return x_reconstructed, x, mask_indices

    def loss(
        self,
        x_reconstructed: torch.Tensor,
        x_original: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:


        x_orig_masked = x_original[mask_indices]
        x_recon_masked = x_reconstructed[mask_indices]

        loss = F.mse_loss(x_recon_masked, x_orig_masked)

        return loss


class StudentModel(nn.Module):
    def __init__(self, encoder, hidden_channels, out_channels, dropout):
        super().__init__()
        self.encoder = encoder

        self.classifier = FraudDetection(hidden_channels, out_channels, dropout=0.1)

    def forward(self, x, edge_index, return_intermediate=False):
        h_v = self.encoder(x, edge_index)

        logits = self.classifier(h_v)

        if return_intermediate:
            return logits, h_v

        return logits
