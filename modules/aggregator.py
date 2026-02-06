import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List, Dict, Optional
import math


class ABTEModule(nn.Module):
    def __init__(
        self,
        tsv_dim: int,
        node_emb_dim: int,
        node_stats_dim: int,
        num_teachers: int,
        num_classes: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.num_teachers = num_teachers
        self.num_classes = num_classes
        # Use LayerNorm to support batch size = 1 during per-node aggregation
        self.tsv_proj = nn.Sequential(
            nn.Linear(tsv_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )
        self.node_proj = nn.Sequential(
            nn.Linear(node_emb_dim + node_stats_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, tsv: Tensor, node_emb: Tensor, node_stats: Tensor, teacher_preds: Tensor, bucket_ids: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        node_feat = torch.cat([node_emb, node_stats], dim=1)
        nf = self.node_proj(node_feat)

        N, T = node_feat.size(0), teacher_preds.size(1)

        if bucket_ids is not None and tsv.dim() == 3:
            num_teachers, num_buckets, tsv_dim = tsv.shape
            bucket_indices = bucket_ids - 1  # [N], 值域 {0, 1, 2, 3}

            bucket_indices_expanded = bucket_indices.unsqueeze(
                # [num_teachers, N, tsv_dim]
                0).unsqueeze(-1).expand(num_teachers, -1, tsv_dim)
            tsv_selected_list = []
            for n in range(N):

                node_tsv = tsv[:, bucket_indices[n].item(), :]
                tsv_selected_list.append(node_tsv)
            tsv_selected = torch.stack(tsv_selected_list, dim=0)
            tf = self.tsv_proj(tsv_selected.view(N * T, -1)).view(N, T, -1)
        else:
            tf = self.tsv_proj(tsv)  # [num_teachers, hidden_dim//4]
            tf = tf.unsqueeze(0).expand(N, -1, -1)

        nf_exp = nf.unsqueeze(1).expand(-1, T, -1)
        comb = torch.cat([nf_exp, tf], dim=2)
        scores = self.scorer(comb.view(N * T, -1)).view(N, T)
        weights = torch.softmax(scores, dim=1)
        agg = torch.bmm(weights.unsqueeze(1), teacher_preds).squeeze(1)
        return agg, weights


class Aggregator(nn.Module):
    def __init__(
        self,
        abte_module: ABTEModule,
        tsv: Tensor,
        eta: float,
        delta: float,
        gamma_dirichlet: float,
        # \Delta_1(F), e.g., s * (Delta_p + Delta_alpha)
        delta_1_F_sensitivity: float,
    ):
        super().__init__()

        self.abte_module = abte_module
        self.register_buffer("tsv", tsv)
        self.delta = delta
        self.eta = eta

        self.gamma = gamma_dirichlet
        self.Delta_1_F = delta_1_F_sensitivity

    def dirichlet_mechanism(self, agg_soft_v):

        alpha_k = self.gamma + (agg_soft_v / self.Delta_1_F) * self.eta
        alpha_k = torch.clamp(alpha_k, min=1e-6)  # 保证正数

        dist = torch.distributions.dirichlet.Dirichlet(alpha_k)
        noisy_label = dist.sample()
        return noisy_label

    def compute_dirichlet_rdp_cost(self, alpha: float) -> float:
        """
        计算 Dirichlet/Gamma 机制 (M_ans) 的单轮 RDP 成本。
        (基于 Balle et al. 2020, Theorem 1 的标准 Gamma 机制 RDP)

        Args:
            alpha: RDP阶数

        Returns:
            rdp_cost: 单次查询的RDP成本
        """
        # 检查约束条件
        if self.eta <= alpha * self.Delta_1_F:
            # 如果 eta 不够大，无法满足 (alpha, delta)-RDP，
            # 返回无穷大成本，RDP 会计器会忽略这个 alpha
            return float("inf")

        # 计算两个 log 项
        term1 = (alpha - 1) * math.log(1 - (self.Delta_1_F / self.eta))
        term2 = -math.log(1 - (alpha * self.Delta_1_F / self.eta))

        # RDP = (gamma / (alpha - 1)) * (term1 + term2)
        rdp_cost = (self.gamma / (alpha - 1)) * (term1 + term2)

        return rdp_cost

    def batch_query_with_abte(
        self,
        node_embeddings: Tensor,  # [N, hidden_dim]
        node_stats: Tensor,  # [N, stats_dim]
        teacher_preds: Tensor,  # [N, S, num_classes]
        bucket_ids: Tensor = None,  # [N]，桶编号，可选
    ) -> Tuple[Tensor, Tensor]:
        """
        使用ABTE模块批量聚合教师预测

        Args:
            node_embeddings: 节点嵌入 [N, hidden_dim]
            node_stats: 节点统计特征 [N, stats_dim]
            teacher_preds: 教师预测（概率格式） [N, S, num_classes]
            bucket_ids: 每个节点对应的桶编号 [N]，每个元素值为 {1, 2, 3, 4} 之一。
                        桶编号根据节点的结构活跃度和特征强度自动计算。
                        如果为None则使用非桶条件化模式

        Returns:
            agg_logits: 聚合后的logits [N, num_classes]
            agg_weights: 注意力权重 [N, S]
        """
        self.abte_module.eval()
        with torch.no_grad():
            agg_logits, agg_weights = self.abte_module(
                self.tsv, node_embeddings, node_stats, teacher_preds, bucket_ids=bucket_ids
            )
        return agg_logits, agg_weights

    def batch_add_noise(
        self,
        agg_soft_labels: Tensor,  # [N, num_classes]
    ) -> Tensor:
        """
        批量对聚合后的软标签添加Dirichlet噪声

        Args:
            agg_soft_labels: 聚合后的软标签（概率格式） [N, num_classes]

        Returns:
            noisy_labels: 加噪后的伪标签 [N, num_classes]
        """
        noisy_labels = []
        for i in range(agg_soft_labels.size(0)):
            agg_soft_v = agg_soft_labels[i]  # [num_classes]
            noisy_label = self.dirichlet_mechanism(agg_soft_v)
            noisy_labels.append(noisy_label)
        return torch.stack(noisy_labels, dim=0)  # [N, num_classes]

    def compute_total_rdp_cost(
        self,
        num_queries: int,
        alphas: Optional[List[float]] = None,
    ) -> Dict:
        """
        计算多次查询的总RDP成本并转换为(epsilon, delta)-DP

        Args:
            num_queries: 查询次数
            alphas: RDP阶数列表，如果为None则使用默认值

        Returns:
            result_dict: 包含RDP核算结果的字典
        """
        from gaussian_accountant.analysis import rdp as privacy_analysis
        from gaussian_accountant.rdp import RDPAccountant

        if alphas is None:
            alphas = RDPAccountant.DEFAULT_ALPHAS

        if num_queries == 0:
            return {
                "total_queries": 0,
                "epsilon_final_rdp": 0.0,
                "best_alpha": None,
                "valid_alphas_count": 0,
                "total_alpha_count": 0,
            }

        # 计算每个alpha阶的单次查询RDP成本
        rdp_costs_total = []
        valid_alphas = []
        total_alpha_count = 0

        for alpha in alphas:
            if alpha > 1:
                total_alpha_count += 1
                rdp_cost_single = self.compute_dirichlet_rdp_cost(alpha)

                # 检查是否有效（不是inf或nan）
                if rdp_cost_single != float("inf") and not math.isnan(rdp_cost_single):
                    # 对于T次独立操作，总RDP成本为T倍单次成本
                    rdp_cost_total = num_queries * rdp_cost_single
                    rdp_costs_total.append(rdp_cost_total)
                    valid_alphas.append(alpha)

        # 转换为(epsilon, delta)-DP
        if len(valid_alphas) > 0 and len(rdp_costs_total) > 0:
            try:
                epsilon_final, best_alpha = privacy_analysis.get_privacy_spent(
                    orders=valid_alphas, rdp=rdp_costs_total, delta=self.delta
                )

                return {
                    "total_queries": num_queries,
                    "eta": self.eta,
                    "epsilon_final_rdp": epsilon_final,
                    "best_alpha": best_alpha,
                    "delta": self.delta,
                    "valid_alphas_count": len(valid_alphas),
                    "total_alpha_count": total_alpha_count,
                    "Delta_1_F": self.Delta_1_F,
                    "gamma": self.gamma,
                }
            except Exception as e:
                return {
                    "total_queries": num_queries,
                    "error": str(e),
                    "epsilon_final_rdp": None,
                    "best_alpha": None,
                }
        else:
            return {
                "total_queries": num_queries,
                "error": "所有alpha阶的RDP成本计算失败（可能eta太小）",
                "epsilon_final_rdp": None,
                "best_alpha": None,
                "valid_alphas_count": 0,
                "total_alpha_count": total_alpha_count,
            }
