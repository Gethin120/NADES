import os
import types
import csv
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from methods.base import FraudDetection
from utils.utils import (
    evaluate,
    get_class_weights,
    split_graph_private_public_pyg,
)
from modules.teacher import TeacherModule
from modules.student import Grace, Infomax, StudentModel, GraphMAE
from modules.gnn_encoder import GNNEncoder
from modules.aggregator import ABTEModule, Aggregator
from trainer import Metrics, Trainer
from sklearn.metrics import roc_auc_score
import GCL.augmentors as A
import GCL.losses as L
from GCL.models import DualBranchContrast
from models.sub_graph import CugraphOverlapPartitioner

augmentor_er = A.EdgeRemoving(pe=0.3)
augmentor_fm = A.FeatureMasking(pf=0.3)


class NADES(FraudDetection):

    def __init__(
        self,
        detection_type: str,
        epochs: int,
        num_neighbor: int,
        num_layers: int,
        batch_size: int,
        lr: float,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        dropout: float,
        logger,
        epsilon: float,
        delta: float,
        max_grad_norm: float,
        patience: int,
        privacy_ratio,
        # Partition method
        partition_method,  # 单一划分方法，默认 "D"
        num_partitions,
        max_overlap_per_node,
        # Teacher
        teacher_epochs,
        teacher_patience,
        # Aggregator
        num_queries,  # 200, 400, 1000
        consistency_weight,  # 1.0, 5.0, 10.0
        ssl_method,  # "infomax", "grace", "graphmae","none"
        # Student
        ssl_epochs: int = 500,
        ssl_patience: int = 20,
        ssl_lr: float = 0.005,
        ssl_grace_tau: float = 0.2,
        dirichlet_gamma: float = 1.0,
        sup_weight: float = 1.0,
        Delta_p_bound: float = 2.0,
        Delta_alpha_bound: float = 1.0,
        evaluate_abte_upper_bound: bool = False,
        metadata=None,
        cache_dir: str = None,  # 缓存目录，默认使用内置路径

        **kwargs,  # Accept any other parameters for compatibility
    ):
        super().__init__(
            detection_type,
            epochs,
            num_neighbor,
            num_layers,
            batch_size,
            lr,
            logger,
            patience,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.delta = delta
        self.dataset = metadata
        self.privacy_ratio = privacy_ratio
        # Partition settings
        self.partition_method = partition_method
        self.S = num_partitions
        self.s_max = max_overlap_per_node
        self.partitioner = CugraphOverlapPartitioner(
            device=self.device,
            train_ratio=0.8,
            val_ratio=0.2,
        )
        # Teacher settings
        self.teacher_epochs = teacher_epochs
        self.teacher_patience = teacher_patience
        self.max_grad_norm = max_grad_norm
        # Aggregator settings
        self.num_queries = num_queries
        self.eta = epsilon
        self.gamma = dirichlet_gamma
        self.Delta_p_bound = Delta_p_bound
        self.Delta_alpha_bound = Delta_alpha_bound
        self.fraud_class_idx = 1 if out_channels > 1 else 0
        self.evaluate_abte_upper_bound = evaluate_abte_upper_bound
        # Cache settings
        if cache_dir is None:
            self.cache_dir = "/home/workspace/Dataset/PDP-GKD"
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        private_data_name = f"{self.dataset}_private_data.pt"
        public_data_name = f"{self.dataset}_public_data.pt"
        self.private_data_path = os.path.join(
            self.cache_dir, private_data_name)
        self.public_data_path = os.path.join(self.cache_dir, public_data_name)

        self.ssl_epochs = ssl_epochs
        self.ssl_patience = ssl_patience
        self.ssl_lr = ssl_lr
        self.ssl_grace_tau = ssl_grace_tau
        self.info = {}
        self.ssl_method = ssl_method.lower()
        if self.ssl_method not in {"infomax", "grace", "graphmae", "none"}:
            raise ValueError(
                f"Unsupported ssl_method '{ssl_method}'. Expected 'infomax', 'grace', 'graphmae', or 'none'."
            )
        self.finetune_patience = patience
        self.lambda_sup = sup_weight
        self.lambda_con = consistency_weight

        self.student_encoder: GNNEncoder = GNNEncoder(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.student: StudentModel = StudentModel(
            encoder=self.student_encoder,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            dropout=self.dropout,
        ).to(self.device)

        # State
        self.private_data: Data | None = None
        self.public_data: Data | None = None
        self.partitions: List[List[int]] | None = None
        self.teachers: List[TeacherModule] = []
        # [num_teachers, num_buckets, 6] for bucket-conditioned TSV
        self.tsv: torch.Tensor | None = None
        self.bucket_thresholds: dict | None = None  # 存储分桶阈值 {tau_deg, tau_fea}
        self.teacher_predictions: torch.Tensor | None = None  # [N_pub, S, C]

    @property
    def detector(self) -> nn.Module:
        return self.student

    def fit(self, data: Data, prefix: str = "nades/") -> nn.Module:
        self.pre_fit(data)
        self.logger.info("(1) Prepare private and public data")
        self._prepare_private_public()

        self.logger.info(f"(2) 使用{self.partition_method}方法进行图划分和子图提取")

        # 构建缓存路径
        # partition_subgraphs_path = os.path.join(
        # self.cache_dir, f"partition_{self.partition_method}_subgraphs.pt"
        # )
        partition_subgraphs_path = None
        # tsv_path = os.path.join(self.cache_dir, f"tsv_{self.partition_method}.pt")
        tsv_path = None
        # 调用单一划分方法
        self.subgraphs = self.partitioner.partition(
            method=self.partition_method,
            data=self.private_data,
            S=self.S,
            s_max=self.s_max,
            resolution=1.0,
            cache_path=partition_subgraphs_path,
            logger=self.logger,
        )

        self.info["partition_method"] = self.partition_method
        self.info["num_subgraphs"] = len(self.subgraphs)
        self.info["max_overlap_per_node"] = self.s_max
        self.logger.info(
            f"划分完成: 方法: {self.partition_method}, 子图数量: {len(self.subgraphs)}, s_max: {self.s_max}"
        )

        self.logger.info("(3) Train teachers with Trainer")
        self.teachers = self._train_teachers_with_trainer(self.subgraphs)

        # 计算桶条件化TSV（在教师训练后）
        self.logger.info("(3.5) Compute bucket-conditioned TSV")
        self.tsv, self.bucket_thresholds = self._compute_bucket_conditioned_tsv(
            self.public_data,
            self.teachers,
            num_buckets=4,
            anchor_ratio=0.1,  # 使用10%的公共节点作为锚点
            logger=self.logger,
        )

        # 计算 TSV 后清理内存
        torch.cuda.empty_cache()

        self.logger.info("(4) Student SSL pretraining on public")
        self._run_ssl_with_trainer_monkey(self.public_data)
        self.logger.info("SSL pretraining completed")

        self.logger.info("(5) Generate privacy-preserving pseudo labels")
        self._generate_pseudo_labels()
        # self._generate_pseudo_labels_mean()
        self.logger.info("Pseudo label generation completed")

        self.logger.info("(6) Semi-supervised fine-tuning with pseudo labels")
        self._run_finetune_with_trainer_monkey(self.public_data)
        self.logger.info("Semi-supervised finetune completed")

        self.logger.info(self.info)
        self._save_info_to_csv()

        return self.student

    def data_loader(self, data: Data, stage, num_neighbors: List[int] = None):
        """Return a dataloader for the given stage."""
        if num_neighbors is None:
            num_neighbors = [self.num_neighbor] * (self.num_layers + 1)
        input_nodes = (
            data.train_mask
            if stage == "train"
            else data.val_mask
            if stage == "val"
            else data.test_mask
        )
        dataloader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=self.batch_size,
            shuffle=(stage == "train"),
            input_nodes=input_nodes,
        )

        return dataloader

    def _save_info_to_csv(self) -> None:
        if not self.info:
            self.logger.warning("self.info 为空，跳过写入 CSV")
            return
        csv_path = os.path.join(self.cache_dir, "pdp_gkd_info.csv")
        try:
            flat_info = self._flatten_info(self.info)
            if not flat_info:
                self.logger.warning("展开后的 self.info 为空，跳过写入 CSV")
                return
            existing_fieldnames: List[str] = []
            existing_rows: List[dict] = []
            if os.path.exists(csv_path):
                with open(csv_path, newline="", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    if reader.fieldnames:
                        existing_fieldnames = list(reader.fieldnames)
                        existing_rows = list(reader)
            fieldnames = existing_fieldnames.copy()
            for key in flat_info.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
            for row in existing_rows:
                for key in fieldnames:
                    row.setdefault(key, "")
            new_row = {key: flat_info.get(key, "") for key in fieldnames}
            existing_rows.append(new_row)
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
            self.logger.info(f"self.info 已写入 {csv_path}")
        except OSError as exc:
            self.logger.error(f"写入 self.info CSV 失败: {exc}")

    def _flatten_info(self, info: dict) -> dict:
        flat: dict = {}

        def _flatten_item(key: str, value):
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    _flatten_item(sub_key, sub_value)
            else:
                flat[key] = self._normalize_value(value)

        for key, value in info.items():
            _flatten_item(key, value)
        return flat

    @staticmethod
    def _normalize_value(value):
        if isinstance(value, (list, tuple, set)):
            return ",".join(map(str, value))
        return value

    def _train_teachers_with_trainer(
        self, subgraphs: List[Data]
    ) -> List[TeacherModule]:
        teachers: List[TeacherModule] = []
        for i, sub in enumerate(subgraphs):
            # 再次验证子图有效性
            if sub.num_nodes < 2:
                self.logger.warning(
                    f"跳过 Teacher {i + 1}：节点数过少 ({sub.num_nodes})"
                )
                continue
            if sub.edge_index.size(1) == 0:
                self.logger.warning(f"跳过 Teacher {i + 1}：没有边")
                continue
            if not hasattr(sub, "train_mask") or not sub.train_mask.any():
                self.logger.warning(f"跳过 Teacher {i + 1}：没有训练节点")
                continue
            # 检查训练集中是否有多个类别
            train_labels = sub.y[sub.train_mask]
            unique_labels = torch.unique(train_labels)
            if len(unique_labels) < 2:
                self.logger.warning(
                    f"跳过 Teacher {i + 1}：训练集只有单一类别 (类别={unique_labels.tolist()})"
                )
                continue

            self.logger.info(f"Teacher {i + 1}/{len(subgraphs)} training:")

            teacher = TeacherModule(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)
            trainer_t = Trainer(
                logger=self.logger,
                patience=self.teacher_patience,
                monitor="val/auc",
                monitor_mode="max",
            )
            optimizer = torch.optim.Adam(teacher.parameters(), lr=self.lr)

            # 猴子补丁：替换 Trainer.step 方法，在 backward 之后、optimizer.step() 之前添加梯度裁剪
            # 使用默认参数捕获循环变量，避免闭包问题
            def teacher_step(
                tr, batch, stage, prefix, teacher_idx=i, subgraph=sub
            ) -> Metrics:
                if stage == "train":
                    tr.optimizer.zero_grad(set_to_none=True)

                grad_state = torch.is_grad_enabled()
                torch.set_grad_enabled(stage == "train")

                # 调用 model.step
                loss, logits_masked, y = teacher.step(
                    batch,
                    stage=stage,
                    global_weights=tr.global_weights,
                )

                torch.set_grad_enabled(grad_state)

                if stage == "train" and loss is not None:
                    # 检查 loss 是否为 NaN 或 Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.error(
                            f"Teacher {teacher_idx + 1} 训练时出现无效损失值: {loss.item()}, "
                            f"子图节点数={subgraph.num_nodes}, 训练节点数={subgraph.train_mask.sum().item()}"
                        )
                        # 跳过这个 batch，不进行反向传播
                        return logits_masked, y, loss

                    with torch.autograd.set_detect_anomaly(True):
                        loss.backward()
                        # 梯度裁剪：在 backward 之后、step 之前
                        torch.nn.utils.clip_grad_norm_(
                            teacher.parameters(), max_norm=self.max_grad_norm
                        )
                        tr.optimizer.step()

                return logits_masked, y, loss

            trainer_t.step = types.MethodType(teacher_step, trainer_t)

            # 单子图作为 dataloader；mask 内已含 train/val
            trainer_t.fit(
                model=teacher,
                epochs=self.teacher_epochs,
                optimizer=optimizer,
                # train_dataloader=self.data_loader(sub, "train"),
                train_dataloader=[sub],
                # val_dataloader=self.data_loader(sub, "val"),
                val_dataloader=[sub],
                test_dataloader=None,
                checkpoint=True,
                global_weights=get_class_weights(sub).to(self.device),
                prefix="teacher/",
            )
            teachers.append(teacher)

            # 清理 trainer 和 optimizer 以释放内存
            del trainer_t, optimizer
            torch.cuda.empty_cache()

        return teachers

    def _run_ssl_with_trainer_monkey(self, pub: Data) -> None:
        """路由到对应的 SSL 训练方法"""
        self.info["ssl_method"] = self.ssl_method

        method_map = {
            "infomax": self._ssl_train_infomax,
            "grace": self._ssl_train_grace,
            "graphmae": self._ssl_train_graphmae,
            "none": self._skip_ssl_pretraining,
        }

        if self.ssl_method not in method_map:
            raise ValueError(
                f"Unsupported ssl_method '{self.ssl_method}'. "
                f"Expected one of {list(method_map.keys())}."
            )

        method_map[self.ssl_method](pub)

    def _skip_ssl_pretraining(self, pub: Data) -> None:
        """
        消融实验：跳过SSL预训练，编码器保持随机初始化状态
        在生成伪标签时，NHV（节点隐藏向量）将使用随机初始化的编码器对节点特征进行编码
        """
        self.info["ssl_method"] = "none (random initialization)"
        self.logger.info(
            "跳过SSL预训练：编码器保持随机初始化状态，"
            "将使用随机初始化的编码器对节点特征进行编码以生成NHV"
        )
        # 确保编码器在评估模式下（虽然参数是随机的）
        self.student_encoder.eval()
        self.student.eval()

    def _run_ssl_training(
        self,
        model: nn.Module,
        loss_fn,
        train_dataloader,
        val_dataloader,
        scheduler_t_max: int = None,
    ) -> dict:
        """运行 SSL 训练的通用流程"""
        device = self.device
        model = model.to(device)

        trainer_ssl = Trainer(
            logger=self.logger,
            patience=self.ssl_patience,
            monitor="val/loss",
            monitor_mode="min",
        )

        def ssl_loop(tr, dataloader, stage: str, prefix: str):
            model = tr.model
            model.train(stage == "train")
            total_loss = 0.0
            steps = 0

            for batch in dataloader:
                if stage == "train":
                    tr.optimizer.zero_grad(set_to_none=True)

                grad_state = torch.is_grad_enabled()
                torch.set_grad_enabled(stage == "train")

                batch = batch.to(device)
                loss = loss_fn(model, batch, tr)

                torch.set_grad_enabled(grad_state)

                if stage == "train" and loss is not None:
                    loss.backward()
                    tr.optimizer.step()
                    if hasattr(tr, "scheduler") and tr.scheduler is not None:
                        tr.scheduler.step()

                total_loss += float(loss.detach().cpu())
                steps += 1

            avg_loss = total_loss / steps if steps > 0 else 0.0
            return {f"{prefix}{stage}/loss": avg_loss}

        if scheduler_t_max is None:
            scheduler_t_max = self.ssl_epochs

        optimizer = torch.optim.Adam(model.parameters(), lr=self.ssl_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_t_max, eta_min=1e-5
        )

        orig_loop = trainer_ssl.loop
        try:
            trainer_ssl.loop = types.MethodType(ssl_loop, trainer_ssl)
            trainer_ssl.optimizer = optimizer
            trainer_ssl.scheduler = scheduler

            best_metrics = trainer_ssl.fit(
                model=model,
                epochs=self.ssl_epochs,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=None,
                checkpoint=True,
                global_weights=None,
                prefix="ssl/",
            )
            return best_metrics
        finally:
            trainer_ssl.loop = orig_loop

    def _ssl_train_infomax(self, pub: Data) -> None:
        """训练 Infomax SSL 模型"""
        infomax_model = Infomax(
            encoder=self.student_encoder,
            hidden_channels=self.hidden_channels,
        )

        def loss_fn(model, batch, tr):
            pos_z, neg_z, summary_vec = model(batch.x, batch.edge_index)
            return model.loss(pos_z, neg_z, summary_vec)

        best_metrics = self._run_ssl_training(
            model=infomax_model,
            loss_fn=loss_fn,
            train_dataloader=[pub],
            val_dataloader=[pub],
        )
        self.info["ssl_best_metrics"] = best_metrics

    def _ssl_train_grace(self, pub: Data) -> None:
        """训练 GRACE SSL 模型"""
        device = self.device

        contrast_model = DualBranchContrast(
            loss=L.InfoNCE(tau=self.ssl_grace_tau), mode="L2L", intraview_negs=False
        ).to(device)

        grace_model = Grace(
            encoder=self.student_encoder,
            hidden_channels=self.hidden_channels,
            projection_dim=self.hidden_channels,
        )

        def loss_fn(model, batch, tr):
            x_1, edge_index_1, _ = augmentor_er(
                batch.x, batch.edge_index, edge_weight=None
            )
            x_2, edge_index_2, _ = augmentor_fm(
                batch.x, batch.edge_index, edge_weight=None
            )
            z_1 = model(x_1, edge_index_1)
            z_2 = model(x_2, edge_index_2)
            h_1, h_2 = [model.projector(z) for z in (z_1, z_2)]
            return contrast_model(h_1, h_2)

        train_loader = self.data_loader(pub, "train")
        val_loader = self.data_loader(pub, "val")
        steps_per_epoch = len(train_loader)
        scheduler_t_max = (
            steps_per_epoch * self.ssl_epochs
            if steps_per_epoch > 0
            else self.ssl_epochs
        )

        best_metrics = self._run_ssl_training(
            model=grace_model,
            loss_fn=loss_fn,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            scheduler_t_max=scheduler_t_max,
        )
        self.info["ssl_best_metrics"] = best_metrics

    def _ssl_train_graphmae(self, pub: Data) -> None:
        """训练 GraphMAE SSL 模型"""
        device = self.device

        graphmae = GraphMAE(
            encoder=self.student_encoder,
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            mask_rate=0.5,
            decoder_dim=self.hidden_channels,
            replace_with_zero=False,
        )

        def loss_fn(model, batch, tr):
            x_recon, x_orig, mask_idx = model(
                batch.x.to(device), batch.edge_index.to(device)
            )
            return model.loss(x_recon, x_orig, mask_idx)

        data = pub.to(device)
        best_metrics = self._run_ssl_training(
            model=graphmae,
            loss_fn=loss_fn,
            train_dataloader=[data],
            val_dataloader=[data],
        )
        self.info["ssl_best_metrics"] = best_metrics

    def _generate_pseudo_labels(self) -> None:
        """
        初始化聚合器（ABTE模块和Aggregator）、预计算缓存，并生成隐私保护的伪标签
        包括：
        1. 初始化ABTE模块和聚合器
        2. 预计算教师预测缓存和节点统计缓存
        3. 生成隐私保护的伪标签并存储到public_data
        """
        # 初始化ABTE模块
        # TSV形状: [num_teachers, num_buckets, 6] 或 [num_teachers, 6]
        if self.tsv.dim() == 3:
            tsv_dim = self.tsv.shape[2]  # 桶条件化TSV
        else:
            tsv_dim = self.tsv.shape[1]  # 非桶条件化TSV

        self.attn = ABTEModule(
            tsv_dim=tsv_dim,
            node_emb_dim=self.hidden_channels,
            node_stats_dim=2,  # NHV包含 [log(1+deg), ||x_v||_2]，共2个统计特征
            num_teachers=len(self.teachers),
            num_classes=self.out_channels,
            hidden_dim=self.hidden_channels,
        ).to(self.device)

        # 设置隐私参数

        self.delta_1_F = self.s_max * \
            (self.Delta_p_bound + self.Delta_alpha_bound)

        # 初始化聚合器
        self.aggregator = Aggregator(
            abte_module=self.attn,
            tsv=self.tsv,
            eta=self.eta,
            delta=self.delta,
            gamma_dirichlet=self.gamma,
            delta_1_F_sensitivity=self.delta_1_F,
        )

        # 预计算教师预测为概率格式 [N, S, num_classes]，其中S是教师数量
        with torch.no_grad():
            self.teacher_preds_cache = self._predict_public_by_teachers(
                self.public_data, self.teachers
            )
            self.public_data.teacher_preds_cache = self.teacher_preds_cache

        # 预计算所有节点的统计特征 [N, stats_dim]
        self.node_stats_cache = self._compute_node_stats(self.public_data).to(
            self.device
        )
        self.public_data.node_stats = self.node_stats_cache

        # 生成隐私保护的伪标签
        self.pseudo_labels_dict = self._query_pseudo_labels_random_abte(
            self.public_data
        )
        self.logger.info(
            f"Has pseudo and true label: {sum(self.public_data.y[self.public_data.has_pseudo_label] == 0)}: normal, {sum(self.public_data.y[self.public_data.has_pseudo_label] == 1)}: fraud"
        )
        if self.evaluate_abte_upper_bound:
            self._evaluate_abte_upper_bound()

    def _generate_pseudo_labels_mean(self) -> None:
        """
        消融实验：使用教师预测的平均值生成伪标签（不使用ABTE）
        包括：
        1. 预计算教师预测缓存
        2. 预计算节点统计缓存（虽然不使用，但保持一致性）
        3. 生成基于教师平均预测的伪标签并存储到public_data
        """
        # 设置隐私参数
        self.delta_1_F = self.s_max * \
            (self.Delta_p_bound + self.Delta_alpha_bound)

        # 初始化聚合器（仅用于加噪，不需要ABTE模块）
        # 创建一个虚拟的ABTE模块（不会被使用）
        dummy_abte = ABTEModule(
            tsv_dim=self.tsv.shape[1],
            node_emb_dim=self.hidden_channels,
            node_stats_dim=2,  # NHV包含 [log(1+deg), ||x_v||_2]，共2个统计特征
            num_teachers=len(self.teachers),
            num_classes=self.out_channels,
            hidden_dim=self.hidden_channels,
        ).to(self.device)

        self.aggregator = Aggregator(
            abte_module=dummy_abte,
            tsv=self.tsv,
            eta=self.eta,
            delta=self.delta,
            gamma_dirichlet=self.gamma,
            delta_1_F_sensitivity=self.delta_1_F,
        )

        # 预计算教师预测为概率格式 [N, S, num_classes]，其中S是教师数量
        with torch.no_grad():
            self.teacher_preds_cache = self._predict_public_by_teachers(
                self.public_data, self.teachers
            )
            self.public_data.teacher_preds_cache = self.teacher_preds_cache

        # 预计算所有节点的统计特征 [N, stats_dim]（虽然不使用，但保持一致性）
        self.node_stats_cache = self._compute_node_stats(self.public_data).to(
            self.device
        )
        self.public_data.node_stats = self.node_stats_cache

        # 生成基于教师平均预测的伪标签
        self.pseudo_labels_dict = self._query_pseudo_labels_mean(
            self.public_data)
        self.logger.info(
            f"Has pseudo and true label: {sum(self.public_data.y[self.public_data.has_pseudo_label] == 0)}: normal, {sum(self.public_data.y[self.public_data.has_pseudo_label] == 1)}: fraud"
        )

    def _run_finetune_with_trainer_monkey(self, pub: Data) -> None:
        trainer_ft = Trainer(
            logger=self.logger,
            patience=self.finetune_patience,
            monitor="val/auc",
            monitor_mode="max",
        )
        self.info["finetune_patience"] = self.finetune_patience
        # self.info["lambda_sup"] = self.lambda_sup
        # self.info["lambda_con"] = self.lambda_con

        def finetune_loop(tr, dataloader, stage: str, prefix: str) -> Metrics:
            # model 是 StudentModel
            model = tr.model
            device = next(model.parameters()).device
            logits_list = []
            y_list = []

            def finetune_step(tr, batch, stage, prefix):
                if stage == "train":
                    model.train()
                    tr.optimizer.zero_grad(set_to_none=True)
                grad_state = torch.is_grad_enabled()
                torch.set_grad_enabled(stage == "train")
                # --- 1. 学生模型前向传播 ---
                #  model 只在小批次 'batch' 上运行
                pi_v_all_batch, _ = model(
                    batch.x, batch.edge_index, return_intermediate=True
                )

                # 取出"中心"节点的结果（只取有效的输入节点）
                valid_batch_size = B_t_indices.numel()
                log_pi_B = pi_v_all_batch[:valid_batch_size]
                y_B = batch.y[:valid_batch_size]

                # --- 2. 计算监督损失 L_sup ---
                # 只对有伪标签的节点计算监督损失
                sup_loss = None
                if (
                    stage == "train"
                    and hasattr(self.public_data, "has_pseudo_label")
                    and self.public_data.has_pseudo_label is not None
                ):
                    # 获取当前批次中有伪标签的节点
                    batch_has_pseudo = self.public_data.has_pseudo_label[
                        B_t_indices
                    ]  # [B]

                    if batch_has_pseudo.any():
                        # 获取有伪标签的节点的预测和伪标签
                        pseudo_labels_B = self.public_data.pseudo_labels[
                            B_t_indices
                        ]  # [B, K]
                        # [B_pseudo, K]
                        log_pi_pseudo = log_pi_B[batch_has_pseudo]
                        pseudo_labels_pseudo = pseudo_labels_B[
                            batch_has_pseudo
                        ]  # [B_pseudo, K]

                        # 计算双向KL散度
                        sup_loss_1 = F.kl_div(
                            log_pi_pseudo,
                            pseudo_labels_pseudo,
                            reduction="batchmean",
                            log_target=False,
                        )
                        sup_loss_2 = F.kl_div(
                            F.log_softmax(pseudo_labels_pseudo, dim=-1),
                            log_pi_pseudo.exp(),
                            reduction="batchmean",
                            log_target=False,
                        )
                        sup_loss = (sup_loss_1 + sup_loss_2) / 2
                    else:
                        # 如果没有伪标签节点，监督损失为0
                        sup_loss = torch.tensor(
                            0.0, device=device, requires_grad=True)

                # --- 3. 计算 L_con ---
                with torch.no_grad():
                    pi = log_pi_B.exp()  # 复用计算

                #  必须对 *同一个批次* 进行增强
                aug_x, aug_edge_index, _ = augmentor_er(
                    batch.x, batch.edge_index, edge_weight=None
                )
                aug_batch_data = batch.clone()
                aug_batch_data.x = aug_x
                aug_batch_data.edge_index = aug_edge_index

                log_pi_prime_batch, _ = model(
                    aug_batch_data.x,
                    aug_batch_data.edge_index,
                    return_intermediate=True,
                )
                log_pi_prime_B = log_pi_prime_batch[:valid_batch_size]

                con_loss_1 = F.kl_div(
                    log_pi_prime_B, pi, reduction="batchmean", log_target=False
                )
                con_loss_2 = F.kl_div(
                    pi.log(),
                    log_pi_prime_B.exp(),
                    reduction="batchmean",
                    log_target=False,
                )
                con_loss = (con_loss_1 + con_loss_2) / 2

                # --- 4. 总损失与更新 ---
                if sup_loss is not None:
                    loss = self.lambda_sup * sup_loss + self.lambda_con * con_loss
                else:
                    loss = self.lambda_con * con_loss

                torch.set_grad_enabled(grad_state)
                if stage == "train" and loss is not None:
                    with torch.autograd.set_detect_anomaly(True):
                        loss.backward()
                        tr.optimizer.step()

                return log_pi_B.detach(), y_B.detach()

            for batch in dataloader:
                batch = batch.to(device)
                B_t_indices = batch.input_id  # 全局索引
                logits, y = finetune_step(tr, batch, stage, prefix)
                # logits, y = finetune_step_privacy(tr, batch, stage, prefix)
                logits_list.append(logits)
                y_list.append(y)
            logits = torch.cat(logits_list, dim=0)
            y = torch.cat(y_list, dim=0).cpu()

            label_mask = y != -1
            if label_mask.any():
                logits = logits[label_mask]
                y = y[label_mask]

            if stage == "test":
                result = evaluate(y, logits)
                return {
                    f"{prefix}{stage}/{key}": value for key, value in result.items()
                }
            else:
                probs = F.softmax(logits, dim=1)[:, 1].cpu().detach().numpy()
                auc = roc_auc_score(y, probs)
                loss = F.cross_entropy(logits.cpu(), y)
                return {
                    f"{prefix}{stage}/auc": auc,
                    f"{prefix}{stage}/loss": loss.item(),
                }

        orig_loop = trainer_ft.loop
        try:
            trainer_ft.loop = types.MethodType(finetune_loop, trainer_ft)
            optimizer = torch.optim.Adam(
                list(self.student.parameters()),
                lr=self.lr,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )
            trainer_ft.scheduler = scheduler
            best_metrics = trainer_ft.fit(
                model=self.student,
                epochs=self.epochs,
                optimizer=optimizer,
                train_dataloader=self.data_loader(pub, "train"),
                val_dataloader=self.data_loader(pub, "val"),
                test_dataloader=self.data_loader(pub, "test"),
                checkpoint=True,
                global_weights=None,
                prefix="finetune/",
            )
            self.info["finetune_best_metrics"] = best_metrics
        finally:
            trainer_ft.loop = orig_loop

    def _prepare_private_public(self):
        # Try load cached
        if os.path.exists(self.private_data_path) and os.path.exists(
            self.public_data_path
        ):
            try:
                self.private_data = torch.load(self.private_data_path)
                self.public_data = torch.load(
                    self.public_data_path).to(self.device)
                return
            except Exception as e:
                self.logger.warning(f"Load private/public cache failed: {e}")

        # Build and cache
        self.logger.info("Caching private/public split to disk")
        private_data, public_data = split_graph_private_public_pyg(
            self.data,
            privacy_ratio=self.privacy_ratio,
            public_split=(0.7, 0.15, 0.15),
            device=self.device,
        )
        self.private_data = private_data
        self.public_data = public_data
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            torch.save(self.private_data, self.private_data_path)
            torch.save(self.public_data, self.public_data_path)
            self.logger.info("Split saved: private_data.pt, public_data.pt")
        except Exception as e:
            self.logger.warning(f"Failed to save cached splits: {e}")

    def _compute_node_stats(self, pub: Data) -> Tensor:
        # 使用 partitioner 的 _degree_vector 方法
        deg = self.partitioner._degree_vector(pub).float().to(self.device)
        # 计算 log(1+deg_pub(v))，与文档一致
        log_deg = torch.log(1 + deg + 1e-8)

        # 计算特征范数 ||x_v||_2（原始值，不归一化）
        feats = pub.x.to(self.device)
        feat_norm = torch.norm(feats, dim=1)

        # 返回 [log(1+deg), ||x_v||_2]
        return torch.stack([log_deg, feat_norm], dim=1)

    def _compute_bucket_id(self, deg_pub: Tensor, feat_norm: Tensor, tau_deg: float, tau_fea: float) -> Tensor:
        """
        计算节点的桶编号

        Args:
            deg_pub: 公共图上的度数 [N]
            feat_norm: 特征范数 [N]
            tau_deg: 度数的中位数阈值
            tau_fea: 特征强度的中位数阈值

        Returns:
            bucket_ids: 每个节点对应的桶编号 [N]，每个元素值为 {1, 2, 3, 4} 之一。
                        桶编号根据公式 k(v)=1 + 𝟙[g_sa(v)≥τ_deg] + 2·𝟙[g_fs(v)≥τ_fea] 自动计算
        """
        # 计算结构活跃度: g_sa(v) = log(1 + deg_pub(v))
        g_sa = torch.log(1 + deg_pub + 1e-8)
        # 计算特征强度: g_fs(v) = ||x_v||_2
        g_fs = feat_norm

        # 桶编号函数: k(v) = 1 + 𝟙[g_sa(v)≥τ_deg] + 2·𝟙[g_fs(v)≥τ_fea]
        bucket_ids = 1 + (g_sa >= tau_deg).long() + \
            2 * (g_fs >= tau_fea).long()
        return bucket_ids

    def _compute_bucket_conditioned_tsv(
        self,
        public_data: Data,
        teachers: List[TeacherModule],
        num_buckets: int = 4,
        anchor_ratio: float = 0.1,
        logger=None,
    ) -> tuple[Tensor, dict]:
        """
        计算桶条件化TSV（Teacher Specificity Vector）

        根据文档NADES.MD的描述：
        1. 选择公共锚点集 V_A ⊂ V_pub
        2. 基于结构活跃度和特征强度对锚点分桶（2×2，得到4个桶）
        3. 对每个教师、每个桶，基于教师对锚点的预测计算统计量

        Args:
            public_data: 公共图数据
            teachers: 教师模型列表
            num_buckets: 桶数量（默认4）
            anchor_ratio: 锚点比例（默认0.1，即10%）
            logger: 日志记录器

        Returns:
            tsv: TSV张量 [num_teachers, num_buckets, 6]
            bucket_thresholds: 分桶阈值字典 {tau_deg, tau_fea}
        """
        device = self.device
        num_teachers = len(teachers)

        # 1. 选择公共锚点集 V_A ⊂ V_pub
        num_anchor_nodes = max(1, int(public_data.num_nodes * anchor_ratio))
        # 随机选择锚点（也可以使用其他策略，如均匀采样）
        anchor_indices = torch.randperm(public_data.num_nodes, device=device)[
            :num_anchor_nodes]
        anchor_indices = anchor_indices.sort()[0]  # 排序以便后续使用

        if logger:
            logger.info(f"选择 {num_anchor_nodes} 个公共锚点用于TSV计算")

        # 2. 计算锚点的公共语境量
        deg_pub = self.partitioner._degree_vector(
            public_data).float().to(device)
        feat_norm = torch.norm(public_data.x, dim=1).to(device)

        anchor_deg = deg_pub[anchor_indices]
        anchor_feat_norm = feat_norm[anchor_indices]

        # 计算结构活跃度: g_sa(v) = log(1 + deg_pub(v))
        g_sa_anchor = torch.log(1 + anchor_deg + 1e-8)
        # 计算特征强度: g_fs(v) = ||x_v||_2
        g_fs_anchor = anchor_feat_norm

        # 3. 计算中位数阈值
        tau_deg = torch.median(g_sa_anchor).item()
        tau_fea = torch.median(g_fs_anchor).item()

        bucket_thresholds = {"tau_deg": tau_deg, "tau_fea": tau_fea}

        if logger:
            logger.info(f"分桶阈值: tau_deg={tau_deg:.4f}, tau_fea={tau_fea:.4f}")

        # 4. 对锚点分桶
        anchor_bucket_ids = self._compute_bucket_id(
            anchor_deg, anchor_feat_norm, tau_deg, tau_fea
        )

        # 统计每个桶的锚点数量
        bucket_counts = {}
        for b in range(1, num_buckets + 1):
            bucket_counts[b] = (anchor_bucket_ids == b).sum().item()
            if logger:
                logger.info(f"桶 {b} 包含 {bucket_counts[b]} 个锚点")

        # 5. 获取教师对锚点的预测
        anchor_x = public_data.x[anchor_indices].to(device)
        anchor_edge_index = public_data.edge_index.to(device)

        teacher_preds_anchor = []  # 每个教师的预测 [num_anchor, num_classes]
        teacher_logits_anchor = []  # 每个教师的logits [num_anchor, num_classes]

        for teacher in teachers:
            teacher.eval()
            with torch.no_grad():
                # 注意：教师模型需要在整个公共图上进行预测（因为图结构）
                # 但我们只关心锚点的预测
                logits_all = teacher(public_data.x.to(
                    device), public_data.edge_index.to(device))
                logits_anchor = logits_all[anchor_indices]
                probs_anchor = F.softmax(logits_anchor, dim=-1)

                teacher_logits_anchor.append(logits_anchor)
                teacher_preds_anchor.append(probs_anchor)

        # [num_teachers, num_anchor, num_classes]
        teacher_preds_anchor = torch.stack(teacher_preds_anchor, dim=0)
        # [num_teachers, num_anchor, num_classes]
        teacher_logits_anchor = torch.stack(teacher_logits_anchor, dim=0)

        # 6. 对每个教师、每个桶计算统计量
        tsv_list = []

        for teacher_idx in range(num_teachers):
            teacher_tsv_buckets = []

            for bucket_id in range(1, num_buckets + 1):
                # 获取该桶的锚点索引
                bucket_mask = (anchor_bucket_ids == bucket_id)
                bucket_anchor_indices_local = torch.where(bucket_mask)[0]

                if len(bucket_anchor_indices_local) == 0:
                    # 如果桶为空，使用零向量
                    teacher_tsv_buckets.append(torch.zeros(6, device=device))
                    if logger:
                        logger.warning(
                            f"教师 {teacher_idx}, 桶 {bucket_id} 为空，使用零向量")
                    continue

                # 获取该桶内锚点的预测
                # [|V_A^b|, num_classes]
                preds_bucket = teacher_preds_anchor[teacher_idx,
                                                    bucket_anchor_indices_local]
                # [|V_A^b|, num_classes]
                logits_bucket = teacher_logits_anchor[teacher_idx,
                                                      bucket_anchor_indices_local]

                # 计算top-1和top-2概率和logits
                top2_probs, top2_indices = torch.topk(
                    preds_bucket, k=2, dim=1)  # [|V_A^b|, 2]
                p_top1 = top2_probs[:, 0]  # [|V_A^b|]
                p_top2 = top2_probs[:, 1]  # [|V_A^b|]

                # 获取对应的logits
                batch_indices = torch.arange(
                    len(bucket_anchor_indices_local), device=device)
                z_top1 = logits_bucket[batch_indices,
                                       top2_indices[:, 0]]  # [|V_A^b|]
                z_top2 = logits_bucket[batch_indices,
                                       top2_indices[:, 1]]  # [|V_A^b|]

                # 计算统计量
                # 1. 平均熵: μ_H^(i,b) = (1/|V_A^b|) Σ_{a∈V_A^b} H(p_a^(i))
                entropy = -torch.sum(preds_bucket *
                                     torch.log(preds_bucket + 1e-8), dim=1)
                mu_H = entropy.mean()

                # 2. 平均概率间隔: μ_M^(i,b) = (1/|V_A^b|) Σ_{a∈V_A^b} (p_{a,1}^(i) - p_{a,2}^(i))
                mu_M = (p_top1 - p_top2).mean()

                # 3. top-1概率分位数: (q_0.25, q_0.5, q_0.75)
                q_25 = torch.quantile(p_top1, 0.25)
                q_50 = torch.quantile(p_top1, 0.50)
                q_75 = torch.quantile(p_top1, 0.75)

                # 4. 温度尺度proxy: μ_tp^(i,b) = (1/|V_A^b|) Σ_{a∈V_A^b} log(1+exp(z_{a,1}^(i)-z_{a,2}^(i)))
                mu_tp = torch.log(1 + torch.exp(z_top1 - z_top2) + 1e-8).mean()

                # 拼接TSV: [μ_H, μ_M, q_0.25, q_0.5, q_0.75, μ_tp]
                tsv_bucket = torch.stack([mu_H, mu_M, q_25, q_50, q_75, mu_tp])
                teacher_tsv_buckets.append(tsv_bucket)

            # [num_buckets, 6]
            tsv_list.append(torch.stack(teacher_tsv_buckets, dim=0))

        # 最终TSV形状: [num_teachers, num_buckets, 6]
        tsv = torch.stack(tsv_list, dim=0).to(torch.float32)

        if logger:
            logger.info(f"TSV计算完成: shape={tuple(tsv.shape)}")

        return tsv, bucket_thresholds

    def _predict_public_by_teachers(
        self, pub: Data, teachers: List[TeacherModule]
    ) -> Tensor:
        logits = []
        x = pub.x.to(self.device)
        e = pub.edge_index.to(self.device)
        for t in teachers:
            t.eval()
            with torch.no_grad():
                teacher_logits = t(x, e)
                logits.append(teacher_logits)
        return torch.stack(logits, dim=1)

    def _query_pseudo_labels_random_abte(self, pub: Data) -> dict:
        """
        一次性随机选择self.num_queries个节点，批量计算ABTE预测，然后对每个节点分别加噪，使用RDP核算隐私成本并汇总总成本
        仅从有标签的训练节点中选择（标签为0或1），排除未标签节点（标签为-1，如elliptic数据集）
        """

        # 获取训练节点索引
        train_node_indices = pub.train_mask.nonzero(as_tuple=True)[
            0].to(self.device)

        # 获取训练节点的标签
        train_labels = pub.y[train_node_indices].to(self.device)

        # 检查是否有未标签节点（标签为-1）
        # 兼容以前的数据集：如果最小标签值 >= 0，说明所有训练节点都是有标签的（标签为0或1）
        min_label = train_labels.min().item()
        if min_label < 0:
            # 有未标签节点，只选择标签为0或1的节点
            labeled_mask = train_labels >= 0  # 标签为0或1的节点（排除-1）
            labeled_train_indices = train_node_indices[labeled_mask]
            self.logger.info(
                f"训练节点总数: {len(train_node_indices)}, "
                f"有标签节点数: {len(labeled_train_indices)} (标签为0或1), "
                f"未标签节点数: {len(train_node_indices) - len(labeled_train_indices)} (标签为-1)"
            )
        else:
            # 兼容以前的数据集：只有两类标签，所有训练节点都是有标签的
            labeled_train_indices = train_node_indices
            self.logger.info(
                f"从训练节点中{len(train_node_indices)}个节点中一次性随机查询 {self.num_queries} 个节点"
            )

        # 确定要查询的节点数量（从有标签的节点中选择）
        num_nodes_to_query = min(self.num_queries, len(labeled_train_indices))
        if num_nodes_to_query < self.num_queries:
            self.logger.warning(
                f"有标签的训练节点数量 ({len(labeled_train_indices)}) 小于查询数量 ({self.num_queries})，"
                f"将查询所有 {num_nodes_to_query} 个有标签的训练节点"
            )

        # 1. 一次性随机选择self.num_queries个节点（仅从有标签的节点中选择）
        if num_nodes_to_query < len(labeled_train_indices):
            # 随机采样
            perm = torch.randperm(
                len(labeled_train_indices), device=self.device)
            selected_indices_local = perm[:num_nodes_to_query]
            selected_indices = labeled_train_indices[
                selected_indices_local
            ]  # 全局节点索引
        else:
            # 如果节点数不够，使用所有有标签的节点
            selected_indices = labeled_train_indices

        # 学生模型与ABTE模块设置为评估模式
        self.student.eval()
        if hasattr(self, "attn") and self.attn is not None:
            self.attn.eval()

        # 计算所有训练节点的节点统计

        # 2. 批量获取学生模型对选中节点的预测和嵌入
        with torch.no_grad():
            _, h_v_all = self.student(
                pub.x.to(self.device),
                pub.edge_index.to(self.device),
                return_intermediate=True,
            )
            # 批量取选中的节点
            # [num_nodes_to_query, hidden_dim]
            selected_h_v = h_v_all[selected_indices]
            selected_node_stats = self.node_stats_cache[
                selected_indices
            ]  # [num_nodes_to_query, node_stats_dim]
            # 获取教师预测
            selected_teacher_preds = self.teacher_preds_cache[selected_indices].to(
                self.device
            )  # [num_nodes_to_query, S, num_classes]

            # 计算选中节点的桶编号（如果使用桶条件化TSV）
            bucket_ids = None
            if self.bucket_thresholds is not None and self.tsv.dim() == 3:
                # 计算选中节点的度数和特征范数
                deg_pub = self.partitioner._degree_vector(
                    self.public_data).float().to(self.device)
                feat_norm = torch.norm(
                    self.public_data.x, dim=1).to(self.device)

                selected_deg = deg_pub[selected_indices]
                selected_feat_norm = feat_norm[selected_indices]

                # 计算桶编号
                bucket_ids = self._compute_bucket_id(
                    selected_deg,
                    selected_feat_norm,
                    self.bucket_thresholds["tau_deg"],
                    self.bucket_thresholds["tau_fea"],
                ).to(self.device)

            # 使用Aggregator的批量查询方法进行ABTE聚合
            agg_logits_selected, _ = self.aggregator.batch_query_with_abte(
                selected_h_v,
                selected_node_stats,
                selected_teacher_preds,
                bucket_ids=bucket_ids,
            )  # [num_nodes_to_query, num_classes]

            # 转换为概率（软标签）
            agg_soft_labels = F.softmax(
                agg_logits_selected, dim=-1
            )  # [num_nodes_to_query, num_classes]

        # 3. 批量使用Dirichlet机制加噪
        self.logger.info("对选中的节点进行Dirichlet机制加噪")
        noisy_labels = self.aggregator.batch_add_noise(
            agg_soft_labels
        )  # [num_nodes_to_query, num_classes]

        # 保存伪标签和加噪前的软标签
        pseudo_labels_dict = {}
        pre_soft_labels_dict = {}
        for i, node_idx in enumerate(selected_indices):
            node_idx_item = node_idx.item()
            pseudo_labels_dict[node_idx_item] = noisy_labels[i].detach().cpu()
            pre_soft_labels_dict[node_idx_item] = agg_soft_labels[i].detach(
            ).cpu()
        # 4. 使用Aggregator的RDP核算方法
        num_queries = len(pseudo_labels_dict)
        if num_queries > 0:
            rdp_result = self.aggregator.compute_total_rdp_cost(num_queries)
            self.epsilon_final = rdp_result["epsilon_final_rdp"]
            self.info["eta"] = self.eta
            self.info["epsilon_final"] = round(self.epsilon_final, 2).item()
            self.info["num_queries"] = self.num_queries

            if "error" in rdp_result:
                self.logger.warning(f"RDP核算计算失败: {rdp_result['error']}")
                self.logger.info("RDP隐私成本核算:")
                self.logger.info(f"  总查询次数: {num_queries}")
            else:
                self.logger.info("使用RDP核算方法计算隐私成本:")
                self.logger.info(
                    f"  总查询次数: {rdp_result['total_queries']}，每次查询eta: {rdp_result['eta']:.2f}，"
                    f"Delta_1_F: {rdp_result['Delta_1_F']:.2f}，gamma: {rdp_result['gamma']:.2f}"
                )
                self.logger.info(
                    f"  RDP核算后的最终epsilon (delta={rdp_result['delta']}): {rdp_result['epsilon_final_rdp']:.2f}，"
                    f"最优RDP阶数alpha: {rdp_result['best_alpha']:.2f}，"
                    f"有效alpha阶数: {rdp_result['valid_alphas_count']}/{rdp_result['total_alpha_count']}"
                )
                if rdp_result["valid_alphas_count"] < rdp_result["total_alpha_count"]:
                    skipped_count = (
                        rdp_result["total_alpha_count"]
                        - rdp_result["valid_alphas_count"]
                    )
                    self.logger.warning(
                        f"  {skipped_count}个alpha阶返回inf（可能 <= alpha * Delta_1_F），"
                        f"这些alpha阶已被跳过"
                    )

                # 保存RDP核算结果
                self.rdp_accounting_result = rdp_result
        else:
            self.logger.warning("未生成任何伪标签，无法进行RDP核算")

        # 统计加噪前的软标签分布
        if len(pre_soft_labels_dict) > 0:
            fraud_count_pre = 0
            normal_count_pre = 0
            fraud_probs_pre = []
            normal_probs_pre = []
            for node_idx, soft_v in pre_soft_labels_dict.items():
                if isinstance(soft_v, torch.Tensor):
                    fraud_class_idx = min(
                        self.fraud_class_idx, len(soft_v) - 1)
                    fraud_prob = soft_v[fraud_class_idx].item()
                    if fraud_prob > 0.5:
                        fraud_count_pre += 1
                        fraud_probs_pre.append(fraud_prob)
                    else:
                        normal_count_pre += 1
                        normal_probs_pre.append(fraud_prob)
            self.info["加噪前偏向欺诈"] = fraud_count_pre
            total_pre = len(pre_soft_labels_dict)
            self.logger.info("加噪前聚合器软标签分布统计:")
            self.logger.info(
                f"  偏向欺诈: {fraud_count_pre} 个 ({fraud_count_pre / total_pre * 100:.2f}%)偏向正常: {normal_count_pre} 个 ({normal_count_pre / total_pre * 100:.2f}%)"
            )
            if len(fraud_probs_pre) > 0:
                self.logger.info(
                    f"  欺诈标签概率: 均值={sum(fraud_probs_pre) / len(fraud_probs_pre):.4f}, "
                    f"最大={max(fraud_probs_pre):.4f}, 最小={min(fraud_probs_pre):.4f}"
                )
            if len(normal_probs_pre) > 0:
                self.logger.info(
                    f"  正常标签概率: 均值={sum(normal_probs_pre) / len(normal_probs_pre):.4f}, "
                    f"最大={max(normal_probs_pre):.4f}, 最小={min(normal_probs_pre):.4f}"
                )

        # 统计伪标签分布（加噪后）
        if len(pseudo_labels_dict) > 0:
            fraud_count = 0
            normal_count = 0
            fraud_probs = []
            normal_probs = []

            for node_idx, pseudo_label in pseudo_labels_dict.items():
                pseudo_label_tensor = pseudo_label
                if isinstance(pseudo_label_tensor, torch.Tensor):
                    fraud_class_idx = min(
                        self.fraud_class_idx, len(pseudo_label_tensor) - 1
                    )
                    fraud_prob = pseudo_label_tensor[fraud_class_idx].item()

                    if fraud_prob > 0.5:
                        fraud_count += 1
                        fraud_probs.append(fraud_prob)
                    else:
                        normal_count += 1
                        normal_probs.append(fraud_prob)
            self.info["加噪后偏向欺诈"] = fraud_count
            self.logger.info("加噪后伪标签分布统计:")
            self.logger.info(
                f"  偏向欺诈: {fraud_count} 个 ({fraud_count / len(pseudo_labels_dict) * 100:.2f}%)偏向正常: {normal_count} 个 ({normal_count / len(pseudo_labels_dict) * 100:.2f}%)"
            )

            if len(fraud_probs) > 0:
                fraud_prob_mean = sum(fraud_probs) / len(fraud_probs)
                fraud_prob_max = max(fraud_probs)
                fraud_prob_min = min(fraud_probs)
                self.logger.info(
                    f"  欺诈标签概率统计: 均值={fraud_prob_mean:.4f}, "
                    f"最大={fraud_prob_max:.4f}, 最小={fraud_prob_min:.4f}"
                )

            if len(normal_probs) > 0:
                normal_prob_mean = sum(normal_probs) / len(normal_probs)
                normal_prob_max = max(normal_probs)
                normal_prob_min = min(normal_probs)
                self.logger.info(
                    f"  正常标签概率统计: 均值={normal_prob_mean:.4f}, "
                    f"最大={normal_prob_max:.4f}, 最小={normal_prob_min:.4f}"
                )

        # 保存到对象，便于其他位置使用
        self.pre_soft_labels_dict = pre_soft_labels_dict

        # 将伪标签转换为tensor并存储到public_data中
        pseudo_labels_tensor = torch.zeros(
            (pub.num_nodes, self.out_channels),
            device=self.device,
            dtype=torch.float32,
        )
        has_pseudo_label = torch.zeros(
            pub.num_nodes, dtype=torch.bool, device=self.device
        )
        for node_idx, pseudo_label in pseudo_labels_dict.items():
            pseudo_labels_tensor[node_idx] = pseudo_label.to(self.device)
            has_pseudo_label[node_idx] = True
        pub.pseudo_labels = pseudo_labels_tensor
        pub.has_pseudo_label = has_pseudo_label
        self.logger.info(
            f"伪标签已存储到public_data，共 {has_pseudo_label.sum().item()} 个节点有伪标签"
        )

        return pseudo_labels_dict

    def _query_pseudo_labels_mean(self, pub: Data) -> dict:
        """
        消融实验：使用教师预测的平均值生成伪标签（不使用ABTE）
        一次性随机选择self.num_queries个节点，计算教师预测的平均值，然后对每个节点分别加噪
        仅从有标签的训练节点中选择（标签为0或1），排除未标签节点（标签为-1，如elliptic数据集）
        """
        # 获取训练节点索引
        train_node_indices = pub.train_mask.nonzero(as_tuple=True)[
            0].to(self.device)

        # 获取训练节点的标签
        train_labels = pub.y[train_node_indices].to(self.device)

        # 检查是否有未标签节点（标签为-1）
        min_label = train_labels.min().item()
        if min_label < 0:
            # 有未标签节点，只选择标签为0或1的节点
            labeled_mask = train_labels >= 0
            labeled_train_indices = train_node_indices[labeled_mask]
            self.logger.info(
                f"训练节点总数: {len(train_node_indices)}, "
                f"有标签节点数: {len(labeled_train_indices)} (标签为0或1), "
                f"未标签节点数: {len(train_node_indices) - len(labeled_train_indices)} (标签为-1)"
            )
        else:
            labeled_train_indices = train_node_indices
            self.logger.info(
                f"从训练节点中{len(train_node_indices)}个节点中一次性随机查询 {self.num_queries} 个节点"
            )

        # 确定要查询的节点数量（从有标签的节点中选择）
        num_nodes_to_query = min(self.num_queries, len(labeled_train_indices))
        if num_nodes_to_query < self.num_queries:
            self.logger.warning(
                f"有标签的训练节点数量 ({len(labeled_train_indices)}) 小于查询数量 ({self.num_queries})，"
                f"将查询所有 {num_nodes_to_query} 个有标签的训练节点"
            )

        # 1. 一次性随机选择self.num_queries个节点（仅从有标签的节点中选择）
        if num_nodes_to_query < len(labeled_train_indices):
            # 随机采样
            perm = torch.randperm(
                len(labeled_train_indices), device=self.device)
            selected_indices_local = perm[:num_nodes_to_query]
            selected_indices = labeled_train_indices[
                selected_indices_local
            ]  # 全局节点索引
        else:
            # 如果节点数不够，使用所有有标签的节点
            selected_indices = labeled_train_indices

        # 2. 获取教师预测并计算平均值
        with torch.no_grad():
            # 获取选中节点的教师预测 [num_nodes_to_query, S, num_classes]
            selected_teacher_preds = self.teacher_preds_cache[selected_indices].to(
                self.device
            )

            # 计算教师预测的平均值 [num_nodes_to_query, num_classes]
            # selected_teacher_preds: [num_nodes_to_query, S, num_classes]
            # mean along teacher dimension (dim=1)
            mean_soft_labels = selected_teacher_preds.mean(
                dim=1
            )  # [num_nodes_to_query, num_classes]

        # 3. 批量使用Dirichlet机制加噪
        self.logger.info("对选中的节点进行Dirichlet机制加噪（基于教师平均预测）")
        noisy_labels = self.aggregator.batch_add_noise(
            mean_soft_labels
        )  # [num_nodes_to_query, num_classes]

        # 保存伪标签和加噪前的软标签
        pseudo_labels_dict = {}
        pre_soft_labels_dict = {}
        for i, node_idx in enumerate(selected_indices):
            node_idx_item = node_idx.item()
            pseudo_labels_dict[node_idx_item] = noisy_labels[i].detach().cpu()
            pre_soft_labels_dict[node_idx_item] = mean_soft_labels[i].detach(
            ).cpu()

        # 4. 使用Aggregator的RDP核算方法
        num_queries = len(pseudo_labels_dict)
        if num_queries > 0:
            rdp_result = self.aggregator.compute_total_rdp_cost(num_queries)
            self.epsilon_final = rdp_result["epsilon_final_rdp"]
            self.info["eta"] = self.eta
            self.info["epsilon_final"] = round(self.epsilon_final, 2).item()
            self.info["num_queries"] = self.num_queries

            if "error" in rdp_result:
                self.logger.warning(f"RDP核算计算失败: {rdp_result['error']}")
                self.logger.info("RDP隐私成本核算:")
                self.logger.info(f"  总查询次数: {num_queries}")
            else:
                self.logger.info("使用RDP核算方法计算隐私成本（基于教师平均预测）:")
                self.logger.info(
                    f"  总查询次数: {rdp_result['total_queries']}，每次查询eta: {rdp_result['eta']:.2f}，"
                    f"Delta_1_F: {rdp_result['Delta_1_F']:.2f}，gamma: {rdp_result['gamma']:.2f}"
                )
                self.logger.info(
                    f"  RDP核算后的最终epsilon (delta={rdp_result['delta']}): {rdp_result['epsilon_final_rdp']:.2f}，"
                    f"最优RDP阶数alpha: {rdp_result['best_alpha']:.2f}，"
                    f"有效alpha阶数: {rdp_result['valid_alphas_count']}/{rdp_result['total_alpha_count']}"
                )
                if rdp_result["valid_alphas_count"] < rdp_result["total_alpha_count"]:
                    skipped_count = (
                        rdp_result["total_alpha_count"]
                        - rdp_result["valid_alphas_count"]
                    )
                    self.logger.warning(
                        f"  {skipped_count}个alpha阶返回inf（可能 <= alpha * Delta_1_F），"
                        f"这些alpha阶已被跳过"
                    )

                # 保存RDP核算结果
                self.rdp_accounting_result = rdp_result
        else:
            self.logger.warning("未生成任何伪标签，无法进行RDP核算")

        # 统计加噪前的软标签分布
        if len(pre_soft_labels_dict) > 0:
            fraud_count_pre = 0
            normal_count_pre = 0
            fraud_probs_pre = []
            normal_probs_pre = []
            for node_idx, soft_v in pre_soft_labels_dict.items():
                if isinstance(soft_v, torch.Tensor):
                    fraud_class_idx = min(
                        self.fraud_class_idx, len(soft_v) - 1)
                    fraud_prob = soft_v[fraud_class_idx].item()
                    if fraud_prob > 0.5:
                        fraud_count_pre += 1
                        fraud_probs_pre.append(fraud_prob)
                    else:
                        normal_count_pre += 1
                        normal_probs_pre.append(fraud_prob)
            self.info["加噪前偏向欺诈(mean)"] = fraud_count_pre
            total_pre = len(pre_soft_labels_dict)
            self.logger.info("加噪前教师平均预测分布统计:")
            self.logger.info(
                f"  偏向欺诈: {fraud_count_pre} 个 ({fraud_count_pre / total_pre * 100:.2f}%)，"
                f"偏向正常: {normal_count_pre} 个 ({normal_count_pre / total_pre * 100:.2f}%)"
            )
            if len(fraud_probs_pre) > 0:
                self.logger.info(
                    f"  欺诈标签概率: 均值={sum(fraud_probs_pre) / len(fraud_probs_pre):.4f}, "
                    f"最大={max(fraud_probs_pre):.4f}, 最小={min(fraud_probs_pre):.4f}"
                )
            if len(normal_probs_pre) > 0:
                self.logger.info(
                    f"  正常标签概率: 均值={sum(normal_probs_pre) / len(normal_probs_pre):.4f}, "
                    f"最大={max(normal_probs_pre):.4f}, 最小={min(normal_probs_pre):.4f}"
                )

        # 统计伪标签分布（加噪后）
        if len(pseudo_labels_dict) > 0:
            fraud_count = 0
            normal_count = 0
            fraud_probs = []
            normal_probs = []

            for node_idx, pseudo_label in pseudo_labels_dict.items():
                pseudo_label_tensor = pseudo_label
                if isinstance(pseudo_label_tensor, torch.Tensor):
                    fraud_class_idx = min(
                        self.fraud_class_idx, len(pseudo_label_tensor) - 1
                    )
                    fraud_prob = pseudo_label_tensor[fraud_class_idx].item()

                    if fraud_prob > 0.5:
                        fraud_count += 1
                        fraud_probs.append(fraud_prob)
                    else:
                        normal_count += 1
                        normal_probs.append(fraud_prob)
            self.info["加噪后偏向欺诈(mean)"] = fraud_count
            self.logger.info("加噪后伪标签分布统计（基于教师平均预测）:")
            self.logger.info(
                f"  偏向欺诈: {fraud_count} 个 ({fraud_count / len(pseudo_labels_dict) * 100:.2f}%)，"
                f"偏向正常: {normal_count} 个 ({normal_count / len(pseudo_labels_dict) * 100:.2f}%)"
            )

            if len(fraud_probs) > 0:
                fraud_prob_mean = sum(fraud_probs) / len(fraud_probs)
                fraud_prob_max = max(fraud_probs)
                fraud_prob_min = min(fraud_probs)
                self.logger.info(
                    f"  欺诈标签概率统计: 均值={fraud_prob_mean:.4f}, "
                    f"最大={fraud_prob_max:.4f}, 最小={fraud_prob_min:.4f}"
                )

            if len(normal_probs) > 0:
                normal_prob_mean = sum(normal_probs) / len(normal_probs)
                normal_prob_max = max(normal_probs)
                normal_prob_min = min(normal_probs)
                self.logger.info(
                    f"  正常标签概率统计: 均值={normal_prob_mean:.4f}, "
                    f"最大={normal_prob_max:.4f}, 最小={normal_prob_min:.4f}"
                )

        # 保存到对象，便于其他位置使用
        self.pre_soft_labels_dict = pre_soft_labels_dict

        # 将伪标签转换为tensor并存储到public_data中
        pseudo_labels_tensor = torch.zeros(
            (pub.num_nodes, self.out_channels),
            device=self.device,
            dtype=torch.float32,
        )
        has_pseudo_label = torch.zeros(
            pub.num_nodes, dtype=torch.bool, device=self.device
        )
        for node_idx, pseudo_label in pseudo_labels_dict.items():
            pseudo_labels_tensor[node_idx] = pseudo_label.to(self.device)
            has_pseudo_label[node_idx] = True
        pub.pseudo_labels = pseudo_labels_tensor
        pub.has_pseudo_label = has_pseudo_label
        self.logger.info(
            f"伪标签已存储到public_data（基于教师平均预测），共 {has_pseudo_label.sum().item()} 个节点有伪标签"
        )

        return pseudo_labels_dict

    def _evaluate_abte_upper_bound(self):
        """
        使用 ABTE 聚合器在所有公开节点上生成无噪声伪标签，
        并与公开数据的真实标签对比，评估理论上限性能。
        """

        if self.public_data is None:
            self.logger.warning("public_data 未初始化，无法评估 ABTE 理论上限")
            return None

        if not hasattr(self, "aggregator") or self.aggregator is None:
            self.logger.warning("aggregator 未初始化，无法评估 ABTE 理论上限")
            return None

        teacher_preds = getattr(self, "teacher_preds_cache", None)
        node_stats = getattr(self, "node_stats_cache", None)

        if teacher_preds is None or node_stats is None:
            self.logger.warning(
                "缺少 teacher 预测或节点统计缓存，无法评估 ABTE 理论上限"
            )
            return None

        self.student.eval()
        if hasattr(self, "attn") and self.attn is not None:
            self.attn.eval()

        pub = self.public_data
        x = pub.x.to(self.device)
        edge_index = pub.edge_index.to(self.device)

        with torch.no_grad():
            _, node_embeddings = self.student(
                x,
                edge_index,
                return_intermediate=True,
            )

            agg_logits, _ = self.aggregator.batch_query_with_abte(
                node_embeddings,
                node_stats,
                teacher_preds.to(self.device),
            )

            agg_probs = F.softmax(agg_logits, dim=-1)

        labels = pub.y.detach().cpu()
        metrics = evaluate(labels, agg_logits.detach().cpu())

        self.logger.info("ABTE 理论上限评估（使用公开标签，无隐私噪声）: %s", metrics)

        pub.abte_upper_bound_logits = agg_logits.detach().cpu()
        pub.abte_upper_bound_probs = agg_probs.detach().cpu()
        self.abte_upper_bound_metrics = metrics

        return metrics
