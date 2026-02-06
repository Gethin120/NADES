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
        """
        Args:
            tsv: TSV tensor. Bucket-conditioned: [num_teachers, num_buckets, tsv_dim];
                otherwise: [num_teachers, tsv_dim].
            node_emb: Node embeddings [N, node_emb_dim]
            node_stats: Node statistics/features [N, node_stats_dim]
            teacher_preds: Teacher predictions [N, num_teachers, num_classes]
            bucket_ids: Bucket ID per node [N], each in {1, 2, 3, 4}.
                Buckets are computed from structural activity and feature strength:
                - bucket 1: low activity + low feature strength
                - bucket 2: high activity + low feature strength
                - bucket 3: low activity + high feature strength
                - bucket 4: high activity + high feature strength
                If None, uses the non-bucket-conditioned mode.
        """
        node_feat = torch.cat([node_emb, node_stats], dim=1)
        nf = self.node_proj(node_feat)

        N, T = node_feat.size(0), teacher_preds.size(1)

        # Bucket-conditioned TSV handling
        if bucket_ids is not None and tsv.dim() == 3:
            # Bucket-conditioned mode: tsv shape [num_teachers, num_buckets, tsv_dim]
            # Select the bucket-specific TSV for each node.
            num_teachers, num_buckets, tsv_dim = tsv.shape
            # bucket_ids are 1-indexed; convert to 0-indexed.
            bucket_indices = bucket_ids - 1  # [N], values in {0, 1, 2, 3}
            # Select bucket-specific TSV per node.
            # (A gather-based implementation is possible; a simple loop is used here.)
            bucket_indices_expanded = bucket_indices.unsqueeze(
                0
            ).unsqueeze(-1).expand(num_teachers, -1, tsv_dim)
            tsv_selected_list = []
            for n in range(N):
                # For node n, select each teacher's TSV in bucket bucket_indices[n].
                node_tsv = tsv[:, bucket_indices[n].item(), :]
                tsv_selected_list.append(node_tsv)
            # [N, num_teachers, tsv_dim]
            tsv_selected = torch.stack(tsv_selected_list, dim=0)
            # Project TSV: [N, num_teachers, hidden_dim//4]
            tf = self.tsv_proj(tsv_selected.view(N * T, -1)).view(N, T, -1)
        else:
            # Non-bucket-conditioned mode: tsv shape [num_teachers, tsv_dim]
            tf = self.tsv_proj(tsv)  # [num_teachers, hidden_dim//4]
            # [N, num_teachers, hidden_dim//4]
            tf = tf.unsqueeze(0).expand(N, -1, -1)

        # [N, num_teachers, hidden_dim//2]
        nf_exp = nf.unsqueeze(1).expand(-1, T, -1)
        # [N, num_teachers, hidden_dim//2 + hidden_dim//4]
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
        # Privacy budget parameters
        self.delta = delta
        self.eta = eta

        # --- Mechanism parameters ---
        self.gamma = gamma_dirichlet
        self.Delta_1_F = delta_1_F_sensitivity

    def dirichlet_mechanism(self, agg_soft_v):
        """
        Generate a noisy pseudo label via the Dirichlet mechanism.

        Args:
            agg_soft_v: Aggregated soft label [num_classes]

        Returns:
            noisy_label: Noisy pseudo label [num_classes]
        """

        alpha_k = self.gamma + (agg_soft_v / self.Delta_1_F) * self.eta
        alpha_k = torch.clamp(alpha_k, min=1e-6)  # ensure positivity

        dist = torch.distributions.dirichlet.Dirichlet(alpha_k)
        noisy_label = dist.sample()
        return noisy_label

    def compute_dirichlet_rdp_cost(self, alpha: float) -> float:
        """
        Compute the per-query RDP cost of the Dirichlet/Gamma mechanism (M_ans).
        (Based on the standard Gamma mechanism RDP in Balle et al., 2020, Theorem 1.)

        Args:
            alpha: RDP order

        Returns:
            rdp_cost: RDP cost for a single query
        """
        # Constraint check
        if self.eta <= alpha * self.Delta_1_F:
            # If eta is too small, we cannot satisfy (alpha, delta)-RDP.
            # Return infinite cost; the accountant will ignore this alpha.
            return float("inf")

        # Two log terms
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
        bucket_ids: Tensor = None,  # [N], optional bucket IDs
    ) -> Tuple[Tensor, Tensor]:
        """
        Batch-aggregate teacher predictions using ABTE.

        Args:
            node_embeddings: Node embeddings [N, hidden_dim]
            node_stats: Node statistics/features [N, stats_dim]
            teacher_preds: Teacher predictions (probabilities) [N, S, num_classes]
            bucket_ids: Bucket ID per node [N], each in {1, 2, 3, 4}.
                If None, uses the non-bucket-conditioned mode.

        Returns:
            agg_logits: Aggregated logits [N, num_classes]
            agg_weights: Attention weights [N, S]
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
        Add Dirichlet noise to aggregated soft labels in batch.

        Args:
            agg_soft_labels: Aggregated soft labels (probabilities) [N, num_classes]

        Returns:
            noisy_labels: Noisy pseudo labels [N, num_classes]
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
        Compute total RDP cost over multiple queries and convert to (epsilon, delta)-DP.

        Args:
            num_queries: Number of queries
            alphas: RDP orders; if None, uses defaults

        Returns:
            result_dict: Dictionary containing the RDP accounting results
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

        # Per-alpha per-query RDP cost
        rdp_costs_total = []
        valid_alphas = []
        total_alpha_count = 0

        for alpha in alphas:
            if alpha > 1:
                total_alpha_count += 1
                rdp_cost_single = self.compute_dirichlet_rdp_cost(alpha)

                # Validate (not inf/nan)
                if rdp_cost_single != float("inf") and not math.isnan(rdp_cost_single):
                    # For T independent operations, total RDP is T times the single-query cost.
                    rdp_cost_total = num_queries * rdp_cost_single
                    rdp_costs_total.append(rdp_cost_total)
                    valid_alphas.append(alpha)

        # Convert to (epsilon, delta)-DP
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
                "error": "RDP cost computation failed for all alpha orders (eta may be too small)",
                "epsilon_final_rdp": None,
                "best_alpha": None,
                "valid_alphas_count": 0,
                "total_alpha_count": total_alpha_count,
            }
