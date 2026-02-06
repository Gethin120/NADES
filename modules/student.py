import torch.nn as nn
from torch_geometric.utils import dropout_adj
from models.fd import FraudDetection
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # 您可以使用您自己的 GNN 层
from torch_geometric.nn.models import DeepGraphInfomax


class SSLModel(nn.Module):
    """自监督学习模型的基类"""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder  # 传入 GNNEncoder 的实例

    def forward(self, x, edge_index):
        """前向传播，子类需要实现"""
        raise NotImplementedError("Subclasses must implement forward method")

    def test(self, x, edge_index):
        """测试模式"""
        self.encoder.eval()


class Grace(SSLModel):
    """GRACE (Graph Contrastive Learning) 方法"""

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
    """Deep Graph Infomax 方法"""

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
        """
        腐蚀函数：保持特征不变，但随机丢弃 p_drop 比例的边。

        Args:
            x: 节点特征 [N, F]
            edge_index: 边索引 [2, E]
            p_drop: 丢弃边的比例 (例如 0.3)

        Returns:
            x_corrupted (x): 原始特征
            edge_index_corrupted: 边被丢弃后的新 edge_index
        """
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
        """
        腐蚀函数：保持结构不变，但随机遮盖 p_mask 比例的节点特征。
        """
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
        """节点特征破坏函数：随机打乱节点顺序"""
        perm = torch.randperm(x.size(0), device=x.device)
        return x[perm], edge_index

    def _summary(self, z, *args, **kwargs):
        """图级摘要函数：对节点表示取平均并应用 sigmoid"""
        return torch.sigmoid(z.mean(dim=0))

    def forward(self, x, edge_index):
        """
        前向传播，返回正样本、负样本和摘要向量

        Returns:
            pos_z: 正样本表示
            neg_z: 负样本表示
            summary_vec: 图级摘要向量
        """
        return self.dgi_model(x, edge_index)

    def loss(self, pos_z, neg_z, summary_vec):
        """
        计算 Infomax 损失

        Args:
            pos_z: 正样本表示
            neg_z: 负样本表示
            summary_vec: 图级摘要向量

        Returns:
            loss: 损失值
        """
        return self.dgi_model.loss(pos_z, neg_z, summary_vec)


class GraphMAE(SSLModel):
    """
    一个 PyG 实现的 GraphMAE (Masked Autoencoder for Graphs)。

    它包含：
    1. 一个 GNN 编码器 (您要预训练的模型)
    2. 一个 Masking 机制
    3. 一个 GNN 解码器 (用于重建)
    """

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
        """
        DGI/GRACE 风格的 .forward() 和 .loss() 分离

        Returns:
            x_reconstructed (Tensor): 重建的特征
            x_original (Tensor): 原始特征 (用于 loss)
            mask_indices (Tensor): 被遮盖的节点索引 (用于 loss)
        """

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
        """
        计算损失 (只在被遮盖的节点上)
        """

        # 提取被遮盖节点的部分
        x_orig_masked = x_original[mask_indices]
        x_recon_masked = x_reconstructed[mask_indices]

        # 计算 MSE 损失
        # (您也可以使用 F.l1_loss，或带归一化的 MSE)
        loss = F.mse_loss(x_recon_masked, x_orig_masked)

        return loss


class StudentModel(nn.Module):
    def __init__(self, encoder, hidden_channels, out_channels, dropout):
        super().__init__()
        # 1. 加载预训练好的 GNN 编码器
        self.encoder = encoder

        # 2. 定义新的分类头 (f_cls)
        self.classifier = FraudDetection(hidden_channels, out_channels, dropout=0.1)

    def forward(self, x, edge_index, return_intermediate=False):
        # 1. 用编码器提取表示
        h_v = self.encoder(x, edge_index)

        # 2. 用分类头进行预测
        logits = self.classifier(h_v)
        # pi_v = F.log_softmax(logits, dim=-1)  # 输出 log-probabilities

        # 如果聚合器需要 h_v 来计算动态注意力
        if return_intermediate:
            return logits, h_v

        return logits
