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
        partition_method,  # Single partition method, default "D"
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
        cache_dir: str = None,  # Cache directory; defaults to the built-in path

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
        self.bucket_thresholds: dict | None = None  # Bucket thresholds {tau_deg, tau_fea}
        self.teacher_predictions: torch.Tensor | None = None  # [N_pub, S, C]

    @property
    def detector(self) -> nn.Module:
        return self.student

    def fit(self, data: Data, prefix: str = "nades/") -> nn.Module:
        self.pre_fit(data)
        self.logger.info("(1) Prepare private and public data")
        self._prepare_private_public()

        self.logger.info(
            f"(2) Partition graph and extract subgraphs using method '{self.partition_method}'"
        )

        # Build cache paths
        # partition_subgraphs_path = os.path.join(
        # self.cache_dir, f"partition_{self.partition_method}_subgraphs.pt"
        # )
        partition_subgraphs_path = None
        # tsv_path = os.path.join(self.cache_dir, f"tsv_{self.partition_method}.pt")
        tsv_path = None
        # Run the selected partition method
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
            f"Partitioning completed: method={self.partition_method}, "
            f"num_subgraphs={len(self.subgraphs)}, s_max={self.s_max}"
        )

        self.logger.info("(3) Train teachers with Trainer")
        self.teachers = self._train_teachers_with_trainer(self.subgraphs)

        # Compute bucket-conditioned TSV (after teacher training)
        self.logger.info("(3.5) Compute bucket-conditioned TSV")
        self.tsv, self.bucket_thresholds = self._compute_bucket_conditioned_tsv(
            self.public_data,
            self.teachers,
            num_buckets=4,
            anchor_ratio=0.1,  # Use 10% of public nodes as anchors
            logger=self.logger,
        )

        # Free GPU memory after TSV computation
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
            self.logger.warning("self.info is empty; skipping CSV write")
            return
        csv_path = os.path.join(self.cache_dir, "pdp_gkd_info.csv")
        try:
            flat_info = self._flatten_info(self.info)
            if not flat_info:
                self.logger.warning("Flattened self.info is empty; skipping CSV write")
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
            self.logger.info(f"self.info written to {csv_path}")
        except OSError as exc:
            self.logger.error(f"Failed to write self.info to CSV: {exc}")

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
            # Validate subgraph again
            if sub.num_nodes < 2:
                self.logger.warning(
                    f"Skipping Teacher {i + 1}: too few nodes ({sub.num_nodes})"
                )
                continue
            if sub.edge_index.size(1) == 0:
                self.logger.warning(f"Skipping Teacher {i + 1}: no edges")
                continue
            if not hasattr(sub, "train_mask") or not sub.train_mask.any():
                self.logger.warning(f"Skipping Teacher {i + 1}: no training nodes")
                continue
            # Ensure training set has at least two classes
            train_labels = sub.y[sub.train_mask]
            unique_labels = torch.unique(train_labels)
            if len(unique_labels) < 2:
                self.logger.warning(
                    f"Skipping Teacher {i + 1}: training set has only one class "
                    f"(classes={unique_labels.tolist()})"
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

            # Monkey patch: replace Trainer.step to add gradient clipping after backward
            # and before optimizer.step(). Use default args to capture loop vars and
            # avoid closure issues.
            def teacher_step(
                tr, batch, stage, prefix, teacher_idx=i, subgraph=sub
            ) -> Metrics:
                if stage == "train":
                    tr.optimizer.zero_grad(set_to_none=True)

                grad_state = torch.is_grad_enabled()
                torch.set_grad_enabled(stage == "train")

                # Call model.step
                loss, logits_masked, y = teacher.step(
                    batch,
                    stage=stage,
                    global_weights=tr.global_weights,
                )

                torch.set_grad_enabled(grad_state)

                if stage == "train" and loss is not None:
                    # Check loss for NaN or Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.error(
                            f"Teacher {teacher_idx + 1} produced an invalid loss "
                            f"(loss={loss.item()}), subgraph_nodes={subgraph.num_nodes}, "
                            f"train_nodes={subgraph.train_mask.sum().item()}"
                        )
                        # Skip this batch and do not backpropagate.
                        return logits_masked, y, loss

                    with torch.autograd.set_detect_anomaly(True):
                        loss.backward()
                        # Gradient clipping: after backward, before step.
                        torch.nn.utils.clip_grad_norm_(
                            teacher.parameters(), max_norm=self.max_grad_norm
                        )
                        tr.optimizer.step()

                return logits_masked, y, loss

            trainer_t.step = types.MethodType(teacher_step, trainer_t)

            # Single subgraph as a dataloader; masks already contain train/val.
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

            # Delete trainer and optimizer to free memory.
            del trainer_t, optimizer
            torch.cuda.empty_cache()

        return teachers

    def _run_ssl_with_trainer_monkey(self, pub: Data) -> None:
        """Route to the selected SSL training method."""
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
        Ablation: skip SSL pretraining and keep the encoder randomly initialized.
        When generating pseudo labels, NHV (node hidden vectors) are encoded using the
        randomly initialized encoder.
        """
        self.info["ssl_method"] = "none (random initialization)"
        self.logger.info(
            "Skipping SSL pretraining: encoder remains randomly initialized; "
            "NHV will be generated by encoding node features with the random encoder."
        )
        # Ensure eval mode (even though weights are random).
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
        """Common routine for running SSL training."""
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
        """Train the Infomax SSL model."""
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
        """Train the GRACE SSL model."""
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
        """Train the GraphMAE SSL model."""
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
        Initialize the aggregator (ABTE module + Aggregator), precompute caches,
        and generate privacy-preserving pseudo labels.

        Steps:
        1) Initialize ABTE module and aggregator
        2) Precompute teacher-prediction cache and node-statistics cache
        3) Generate privacy-preserving pseudo labels and store them in public_data
        """
        # Initialize ABTE module.
        # TSV shape: [num_teachers, num_buckets, 6] or [num_teachers, 6]
        if self.tsv.dim() == 3:
            tsv_dim = self.tsv.shape[2]  # Bucket-conditioned TSV
        else:
            tsv_dim = self.tsv.shape[1]  # Non-bucket-conditioned TSV

        self.attn = ABTEModule(
            tsv_dim=tsv_dim,
            node_emb_dim=self.hidden_channels,
            node_stats_dim=2,  # NHV includes [log(1+deg), ||x_v||_2]
            num_teachers=len(self.teachers),
            num_classes=self.out_channels,
            hidden_dim=self.hidden_channels,
        ).to(self.device)

        # Privacy parameters

        self.delta_1_F = self.s_max * \
            (self.Delta_p_bound + self.Delta_alpha_bound)

        # Initialize aggregator
        self.aggregator = Aggregator(
            abte_module=self.attn,
            tsv=self.tsv,
            eta=self.eta,
            delta=self.delta,
            gamma_dirichlet=self.gamma,
            delta_1_F_sensitivity=self.delta_1_F,
        )

        # Precompute teacher predictions as logits [N, S, num_classes], where S is
        # the number of teachers.
        with torch.no_grad():
            self.teacher_preds_cache = self._predict_public_by_teachers(
                self.public_data, self.teachers
            )
            self.public_data.teacher_preds_cache = self.teacher_preds_cache

        # Precompute per-node statistics [N, stats_dim].
        self.node_stats_cache = self._compute_node_stats(self.public_data).to(
            self.device
        )
        self.public_data.node_stats = self.node_stats_cache

        # Generate privacy-preserving pseudo labels.
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
        Ablation: generate pseudo labels using the mean of teacher predictions
        (without ABTE).

        Steps:
        1) Precompute teacher-prediction cache
        2) Precompute node-statistics cache (kept for consistency)
        3) Generate pseudo labels from mean teacher predictions and store them in public_data
        """
        # Privacy parameters
        self.delta_1_F = self.s_max * \
            (self.Delta_p_bound + self.Delta_alpha_bound)

        # Initialize aggregator (noise only; ABTE is not used here).
        # Create a dummy ABTE module (won't be used).
        dummy_abte = ABTEModule(
            tsv_dim=self.tsv.shape[1],
            node_emb_dim=self.hidden_channels,
            node_stats_dim=2,  # NHV includes [log(1+deg), ||x_v||_2]
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

        # Precompute teacher predictions as logits [N, S, num_classes], where S is
        # the number of teachers.
        with torch.no_grad():
            self.teacher_preds_cache = self._predict_public_by_teachers(
                self.public_data, self.teachers
            )
            self.public_data.teacher_preds_cache = self.teacher_preds_cache

        # Precompute per-node statistics [N, stats_dim] (kept for consistency).
        self.node_stats_cache = self._compute_node_stats(self.public_data).to(
            self.device
        )
        self.public_data.node_stats = self.node_stats_cache

        # Generate pseudo labels from mean teacher predictions.
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
            # model is a StudentModel
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
                # --- 1) Forward pass (student) ---
                # model runs on the mini-batch only.
                pi_v_all_batch, _ = model(
                    batch.x, batch.edge_index, return_intermediate=True
                )

                # Extract "center" node outputs (only the valid input nodes).
                valid_batch_size = B_t_indices.numel()
                log_pi_B = pi_v_all_batch[:valid_batch_size]
                y_B = batch.y[:valid_batch_size]

                # --- 2) Supervised loss (L_sup) ---
                # Compute supervision only for nodes that have pseudo labels.
                sup_loss = None
                if (
                    stage == "train"
                    and hasattr(self.public_data, "has_pseudo_label")
                    and self.public_data.has_pseudo_label is not None
                ):
                    # Nodes in the batch that have pseudo labels
                    batch_has_pseudo = self.public_data.has_pseudo_label[
                        B_t_indices
                    ]  # [B]

                    if batch_has_pseudo.any():
                        # Predictions and pseudo labels for pseudo-labeled nodes
                        pseudo_labels_B = self.public_data.pseudo_labels[
                            B_t_indices
                        ]  # [B, K]
                        # [B_pseudo, K]
                        log_pi_pseudo = log_pi_B[batch_has_pseudo]
                        pseudo_labels_pseudo = pseudo_labels_B[
                            batch_has_pseudo
                        ]  # [B_pseudo, K]

                        # Symmetric KL divergence
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
                        # No pseudo-labeled nodes -> zero supervised loss
                        sup_loss = torch.tensor(
                            0.0, device=device, requires_grad=True)

                # --- 3) Consistency loss (L_con) ---
                with torch.no_grad():
                    pi = log_pi_B.exp()  # Reuse computed probabilities

                # Apply augmentation on the same batch
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

                # --- 4) Total loss and update ---
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
                B_t_indices = batch.input_id  # Global indices
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
        # Use partitioner's _degree_vector method
        deg = self.partitioner._degree_vector(pub).float().to(self.device)
        # Compute log(1 + deg_pub(v)), consistent with the docs
        log_deg = torch.log(1 + deg + 1e-8)

        # Compute feature norm ||x_v||_2 (raw, not normalized)
        feats = pub.x.to(self.device)
        feat_norm = torch.norm(feats, dim=1)

        # Return [log(1+deg), ||x_v||_2]
        return torch.stack([log_deg, feat_norm], dim=1)

    def _compute_bucket_id(self, deg_pub: Tensor, feat_norm: Tensor, tau_deg: float, tau_fea: float) -> Tensor:
        """
        Compute per-node bucket IDs.

        Args:
            deg_pub: Degree on the public graph [N]
            feat_norm: Feature norm [N]
            tau_deg: Median threshold for degree activity
            tau_fea: Median threshold for feature strength

        Returns:
            bucket_ids: Bucket IDs [N], each in {1, 2, 3, 4}.
                Computed as: k(v) = 1 + 1[g_sa(v) >= tau_deg] + 2 * 1[g_fs(v) >= tau_fea]
        """
        # Structural activity: g_sa(v) = log(1 + deg_pub(v))
        g_sa = torch.log(1 + deg_pub + 1e-8)
        # Feature strength: g_fs(v) = ||x_v||_2
        g_fs = feat_norm

        # Bucket function: k(v) = 1 + 1[g_sa(v) >= tau_deg] + 2 * 1[g_fs(v) >= tau_fea]
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
        Compute a bucket-conditioned TSV (Teacher Specificity Vector).

        Following the NADES.MD description:
        1) Choose a public anchor set V_A ⊂ V_pub
        2) Bucket anchors by structural activity and feature strength (2x2 -> 4 buckets)
        3) For each teacher and bucket, compute statistics based on teacher predictions

        Args:
            public_data: Public graph data
            teachers: List of teacher models
            num_buckets: Number of buckets (default: 4)
            anchor_ratio: Anchor ratio (default: 0.1 i.e. 10%)
            logger: Logger

        Returns:
            tsv: TSV tensor [num_teachers, num_buckets, 6]
            bucket_thresholds: Threshold dict {tau_deg, tau_fea}
        """
        device = self.device
        num_teachers = len(teachers)

        # 1) Choose public anchor set V_A ⊂ V_pub
        num_anchor_nodes = max(1, int(public_data.num_nodes * anchor_ratio))
        # Randomly sample anchors (other strategies are possible).
        anchor_indices = torch.randperm(public_data.num_nodes, device=device)[
            :num_anchor_nodes]
        anchor_indices = anchor_indices.sort()[0]  # Sort for convenience

        if logger:
            logger.info(f"Selected {num_anchor_nodes} public anchor nodes for TSV")

        # 2) Compute public context quantities for anchors
        deg_pub = self.partitioner._degree_vector(
            public_data).float().to(device)
        feat_norm = torch.norm(public_data.x, dim=1).to(device)

        anchor_deg = deg_pub[anchor_indices]
        anchor_feat_norm = feat_norm[anchor_indices]

        # Structural activity: g_sa(v) = log(1 + deg_pub(v))
        g_sa_anchor = torch.log(1 + anchor_deg + 1e-8)
        # Feature strength: g_fs(v) = ||x_v||_2
        g_fs_anchor = anchor_feat_norm

        # 3) Compute median thresholds
        tau_deg = torch.median(g_sa_anchor).item()
        tau_fea = torch.median(g_fs_anchor).item()

        bucket_thresholds = {"tau_deg": tau_deg, "tau_fea": tau_fea}

        if logger:
            logger.info(f"Bucket thresholds: tau_deg={tau_deg:.4f}, tau_fea={tau_fea:.4f}")

        # 4) Bucket anchors
        anchor_bucket_ids = self._compute_bucket_id(
            anchor_deg, anchor_feat_norm, tau_deg, tau_fea
        )

        # Count anchors per bucket
        bucket_counts = {}
        for b in range(1, num_buckets + 1):
            bucket_counts[b] = (anchor_bucket_ids == b).sum().item()
            if logger:
                logger.info(f"Bucket {b} contains {bucket_counts[b]} anchors")

        # 5) Get teacher predictions on anchors
        anchor_x = public_data.x[anchor_indices].to(device)
        anchor_edge_index = public_data.edge_index.to(device)

        teacher_preds_anchor = []  # Per-teacher probs [num_anchor, num_classes]
        teacher_logits_anchor = []  # Per-teacher logits [num_anchor, num_classes]

        for teacher in teachers:
            teacher.eval()
            with torch.no_grad():
                # Note: Teacher predicts over the whole public graph (graph structure),
                # but we only use the anchor node outputs.
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

        # 6) Compute stats per teacher and bucket
        tsv_list = []

        for teacher_idx in range(num_teachers):
            teacher_tsv_buckets = []

            for bucket_id in range(1, num_buckets + 1):
                # Anchor indices in this bucket
                bucket_mask = (anchor_bucket_ids == bucket_id)
                bucket_anchor_indices_local = torch.where(bucket_mask)[0]

                if len(bucket_anchor_indices_local) == 0:
                    # Empty bucket -> use zeros
                    teacher_tsv_buckets.append(torch.zeros(6, device=device))
                    if logger:
                        logger.warning(
                            f"Teacher {teacher_idx}, bucket {bucket_id} is empty; using zeros"
                        )
                    continue

                # Predictions within this bucket
                # [|V_A^b|, num_classes]
                preds_bucket = teacher_preds_anchor[teacher_idx,
                                                    bucket_anchor_indices_local]
                # [|V_A^b|, num_classes]
                logits_bucket = teacher_logits_anchor[teacher_idx,
                                                      bucket_anchor_indices_local]

                # Top-1/Top-2 probabilities and logits
                top2_probs, top2_indices = torch.topk(
                    preds_bucket, k=2, dim=1)  # [|V_A^b|, 2]
                p_top1 = top2_probs[:, 0]  # [|V_A^b|]
                p_top2 = top2_probs[:, 1]  # [|V_A^b|]

                # Corresponding logits
                batch_indices = torch.arange(
                    len(bucket_anchor_indices_local), device=device)
                z_top1 = logits_bucket[batch_indices,
                                       top2_indices[:, 0]]  # [|V_A^b|]
                z_top2 = logits_bucket[batch_indices,
                                       top2_indices[:, 1]]  # [|V_A^b|]

                # Stats:
                # 1) Mean entropy: μ_H^(i,b) = (1/|V_A^b|) Σ_{a∈V_A^b} H(p_a^(i))
                entropy = -torch.sum(preds_bucket *
                                     torch.log(preds_bucket + 1e-8), dim=1)
                mu_H = entropy.mean()

                # 2) Mean probability margin:
                #    μ_M^(i,b) = (1/|V_A^b|) Σ_{a∈V_A^b} (p_{a,1}^(i) - p_{a,2}^(i))
                mu_M = (p_top1 - p_top2).mean()

                # 3) Top-1 probability quantiles: (q_0.25, q_0.5, q_0.75)
                q_25 = torch.quantile(p_top1, 0.25)
                q_50 = torch.quantile(p_top1, 0.50)
                q_75 = torch.quantile(p_top1, 0.75)

                # 4) Temperature-scale proxy:
                #    μ_tp^(i,b) = (1/|V_A^b|) Σ_{a∈V_A^b} log(1+exp(z_{a,1}^(i)-z_{a,2}^(i)))
                mu_tp = torch.log(1 + torch.exp(z_top1 - z_top2) + 1e-8).mean()

                # TSV: [μ_H, μ_M, q_0.25, q_0.5, q_0.75, μ_tp]
                tsv_bucket = torch.stack([mu_H, mu_M, q_25, q_50, q_75, mu_tp])
                teacher_tsv_buckets.append(tsv_bucket)

            # [num_buckets, 6]
            tsv_list.append(torch.stack(teacher_tsv_buckets, dim=0))

        # Final TSV shape: [num_teachers, num_buckets, 6]
        tsv = torch.stack(tsv_list, dim=0).to(torch.float32)

        if logger:
            logger.info(f"TSV computation completed: shape={tuple(tsv.shape)}")

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
        Randomly select `self.num_queries` nodes (one shot), batch-compute ABTE
        predictions, then add noise per node. Uses RDP accounting to compute and
        aggregate the privacy cost.

        Nodes are selected from labeled training nodes only (labels 0/1), excluding
        unlabeled nodes (label -1, e.g. Elliptic dataset).
        """

        # Training node indices
        train_node_indices = pub.train_mask.nonzero(as_tuple=True)[
            0].to(self.device)

        # Training labels
        train_labels = pub.y[train_node_indices].to(self.device)

        # Check for unlabeled nodes (label -1). Backward-compatible: if min label
        # >= 0 then all training nodes are labeled (0/1).
        min_label = train_labels.min().item()
        if min_label < 0:
            # Has unlabeled nodes -> select only labeled nodes (0/1).
            labeled_mask = train_labels >= 0  # exclude -1
            labeled_train_indices = train_node_indices[labeled_mask]
            self.logger.info(
                f"Total train nodes: {len(train_node_indices)}, "
                f"labeled: {len(labeled_train_indices)} (label 0/1), "
                f"unlabeled: {len(train_node_indices) - len(labeled_train_indices)} (label -1)"
            )
        else:
            # All training nodes are labeled.
            labeled_train_indices = train_node_indices
            self.logger.info(
                f"Randomly querying {self.num_queries} nodes from {len(train_node_indices)} train nodes"
            )

        # Number of nodes to query (from labeled nodes)
        num_nodes_to_query = min(self.num_queries, len(labeled_train_indices))
        if num_nodes_to_query < self.num_queries:
            self.logger.warning(
                f"Labeled train nodes ({len(labeled_train_indices)}) < requested queries ({self.num_queries}); "
                f"querying all {num_nodes_to_query} labeled train nodes"
            )

        # 1) Randomly select nodes (from labeled nodes only)
        if num_nodes_to_query < len(labeled_train_indices):
            # Random sampling
            perm = torch.randperm(
                len(labeled_train_indices), device=self.device)
            selected_indices_local = perm[:num_nodes_to_query]
            selected_indices = labeled_train_indices[
                selected_indices_local
            ]  # Global node indices
        else:
            # Not enough nodes -> use all labeled nodes
            selected_indices = labeled_train_indices

        # Set student model and ABTE module to eval mode
        self.student.eval()
        if hasattr(self, "attn") and self.attn is not None:
            self.attn.eval()

        # 2) Batch-get student embeddings for the selected nodes
        with torch.no_grad():
            _, h_v_all = self.student(
                pub.x.to(self.device),
                pub.edge_index.to(self.device),
                return_intermediate=True,
            )
            # Gather selected nodes
            # [num_nodes_to_query, hidden_dim]
            selected_h_v = h_v_all[selected_indices]
            selected_node_stats = self.node_stats_cache[
                selected_indices
            ]  # [num_nodes_to_query, node_stats_dim]
            # Teacher predictions
            selected_teacher_preds = self.teacher_preds_cache[selected_indices].to(
                self.device
            )  # [num_nodes_to_query, S, num_classes]

            # Bucket IDs for selected nodes (if using bucket-conditioned TSV)
            bucket_ids = None
            if self.bucket_thresholds is not None and self.tsv.dim() == 3:
                # Degree and feature norms for selected nodes
                deg_pub = self.partitioner._degree_vector(
                    self.public_data).float().to(self.device)
                feat_norm = torch.norm(
                    self.public_data.x, dim=1).to(self.device)

                selected_deg = deg_pub[selected_indices]
                selected_feat_norm = feat_norm[selected_indices]

                # Bucket IDs
                bucket_ids = self._compute_bucket_id(
                    selected_deg,
                    selected_feat_norm,
                    self.bucket_thresholds["tau_deg"],
                    self.bucket_thresholds["tau_fea"],
                ).to(self.device)

            # ABTE aggregation via aggregator's batch query
            agg_logits_selected, _ = self.aggregator.batch_query_with_abte(
                selected_h_v,
                selected_node_stats,
                selected_teacher_preds,
                bucket_ids=bucket_ids,
            )  # [num_nodes_to_query, num_classes]

            # Convert to probabilities (soft labels)
            agg_soft_labels = F.softmax(
                agg_logits_selected, dim=-1
            )  # [num_nodes_to_query, num_classes]

        # 3) Add noise in batch via Dirichlet mechanism
        self.logger.info("Adding Dirichlet noise to selected nodes")
        noisy_labels = self.aggregator.batch_add_noise(
            agg_soft_labels
        )  # [num_nodes_to_query, num_classes]

        # Store pseudo labels and pre-noise soft labels
        pseudo_labels_dict = {}
        pre_soft_labels_dict = {}
        for i, node_idx in enumerate(selected_indices):
            node_idx_item = node_idx.item()
            pseudo_labels_dict[node_idx_item] = noisy_labels[i].detach().cpu()
            pre_soft_labels_dict[node_idx_item] = agg_soft_labels[i].detach(
            ).cpu()
        # 4) RDP accounting
        num_queries = len(pseudo_labels_dict)
        if num_queries > 0:
            rdp_result = self.aggregator.compute_total_rdp_cost(num_queries)
            self.epsilon_final = rdp_result["epsilon_final_rdp"]
            self.info["eta"] = self.eta
            self.info["epsilon_final"] = round(self.epsilon_final, 2).item()
            self.info["num_queries"] = self.num_queries

            if "error" in rdp_result:
                self.logger.warning(f"RDP accounting failed: {rdp_result['error']}")
                self.logger.info("RDP privacy cost accounting:")
                self.logger.info(f"  Total queries: {num_queries}")
            else:
                self.logger.info("Privacy cost via RDP accounting:")
                self.logger.info(
                    f"  Total queries: {rdp_result['total_queries']}, eta per query: {rdp_result['eta']:.2f}, "
                    f"Delta_1_F: {rdp_result['Delta_1_F']:.2f}, gamma: {rdp_result['gamma']:.2f}"
                )
                self.logger.info(
                    f"  Final epsilon via RDP (delta={rdp_result['delta']}): {rdp_result['epsilon_final_rdp']:.2f}, "
                    f"best alpha: {rdp_result['best_alpha']:.2f}, "
                    f"valid alphas: {rdp_result['valid_alphas_count']}/{rdp_result['total_alpha_count']}"
                )
                if rdp_result["valid_alphas_count"] < rdp_result["total_alpha_count"]:
                    skipped_count = (
                        rdp_result["total_alpha_count"]
                        - rdp_result["valid_alphas_count"]
                    )
                    self.logger.warning(
                        f"  {skipped_count} alpha orders returned inf (possibly <= alpha * Delta_1_F); skipped"
                    )

                # Save RDP accounting results
                self.rdp_accounting_result = rdp_result
        else:
            self.logger.warning("No pseudo labels generated; cannot run RDP accounting")

        # Pre-noise soft-label distribution
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
            self.info["pre_noise_fraud_leaning"] = fraud_count_pre
            total_pre = len(pre_soft_labels_dict)
            self.logger.info("Pre-noise aggregator soft-label distribution:")
            self.logger.info(
                f"  Fraud-leaning: {fraud_count_pre} ({fraud_count_pre / total_pre * 100:.2f}%), "
                f"normal-leaning: {normal_count_pre} ({normal_count_pre / total_pre * 100:.2f}%)"
            )
            if len(fraud_probs_pre) > 0:
                self.logger.info(
                    f"  Fraud-class prob (fraud-leaning) stats: mean={sum(fraud_probs_pre) / len(fraud_probs_pre):.4f}, "
                    f"max={max(fraud_probs_pre):.4f}, min={min(fraud_probs_pre):.4f}"
                )
            if len(normal_probs_pre) > 0:
                self.logger.info(
                    f"  Fraud-class prob (normal-leaning) stats: mean={sum(normal_probs_pre) / len(normal_probs_pre):.4f}, "
                    f"max={max(normal_probs_pre):.4f}, min={min(normal_probs_pre):.4f}"
                )

        # Post-noise pseudo-label distribution
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
            self.info["post_noise_fraud_leaning"] = fraud_count
            self.logger.info("Post-noise pseudo-label distribution:")
            self.logger.info(
                f"  Fraud-leaning: {fraud_count} ({fraud_count / len(pseudo_labels_dict) * 100:.2f}%), "
                f"normal-leaning: {normal_count} ({normal_count / len(pseudo_labels_dict) * 100:.2f}%)"
            )

            if len(fraud_probs) > 0:
                fraud_prob_mean = sum(fraud_probs) / len(fraud_probs)
                fraud_prob_max = max(fraud_probs)
                fraud_prob_min = min(fraud_probs)
                self.logger.info(
                    f"  Fraud-class prob (fraud-leaning) stats: mean={fraud_prob_mean:.4f}, "
                    f"max={fraud_prob_max:.4f}, min={fraud_prob_min:.4f}"
                )

            if len(normal_probs) > 0:
                normal_prob_mean = sum(normal_probs) / len(normal_probs)
                normal_prob_max = max(normal_probs)
                normal_prob_min = min(normal_probs)
                self.logger.info(
                    f"  Fraud-class prob (normal-leaning) stats: mean={normal_prob_mean:.4f}, "
                    f"max={normal_prob_max:.4f}, min={normal_prob_min:.4f}"
                )

        # Store for later use
        self.pre_soft_labels_dict = pre_soft_labels_dict

        # Convert pseudo labels to tensors and store in public_data
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
            f"Pseudo labels saved to public_data; {has_pseudo_label.sum().item()} nodes have pseudo labels"
        )

        return pseudo_labels_dict

    def _query_pseudo_labels_mean(self, pub: Data) -> dict:
        """
        Ablation: generate pseudo labels using the mean of teacher predictions
        (without ABTE).

        Randomly select `self.num_queries` nodes (one shot), compute the mean of
        teacher predictions, then add noise per node. Nodes are selected from labeled
        training nodes only (labels 0/1), excluding unlabeled nodes (label -1, e.g.
        Elliptic dataset).
        """
        # Training node indices
        train_node_indices = pub.train_mask.nonzero(as_tuple=True)[
            0].to(self.device)

        # Training labels
        train_labels = pub.y[train_node_indices].to(self.device)

        # Check for unlabeled nodes (label -1)
        min_label = train_labels.min().item()
        if min_label < 0:
            # Has unlabeled nodes -> select only labeled nodes (0/1)
            labeled_mask = train_labels >= 0
            labeled_train_indices = train_node_indices[labeled_mask]
            self.logger.info(
                f"Total train nodes: {len(train_node_indices)}, "
                f"labeled: {len(labeled_train_indices)} (label 0/1), "
                f"unlabeled: {len(train_node_indices) - len(labeled_train_indices)} (label -1)"
            )
        else:
            labeled_train_indices = train_node_indices
            self.logger.info(
                f"Randomly querying {self.num_queries} nodes from {len(train_node_indices)} train nodes"
            )

        # Number of nodes to query (from labeled nodes)
        num_nodes_to_query = min(self.num_queries, len(labeled_train_indices))
        if num_nodes_to_query < self.num_queries:
            self.logger.warning(
                f"Labeled train nodes ({len(labeled_train_indices)}) < requested queries ({self.num_queries}); "
                f"querying all {num_nodes_to_query} labeled train nodes"
            )

        # 1) Randomly select nodes (from labeled nodes only)
        if num_nodes_to_query < len(labeled_train_indices):
            # Random sampling
            perm = torch.randperm(
                len(labeled_train_indices), device=self.device)
            selected_indices_local = perm[:num_nodes_to_query]
            selected_indices = labeled_train_indices[
                selected_indices_local
            ]  # Global node indices
        else:
            # Not enough nodes -> use all labeled nodes
            selected_indices = labeled_train_indices

        # 2) Mean teacher predictions
        with torch.no_grad():
            # Teacher predictions for selected nodes: [num_nodes_to_query, S, num_classes]
            selected_teacher_preds = self.teacher_preds_cache[selected_indices].to(
                self.device
            )

            # Mean over teachers: [num_nodes_to_query, num_classes]
            # selected_teacher_preds: [num_nodes_to_query, S, num_classes]
            # mean along teacher dimension (dim=1)
            mean_soft_labels = selected_teacher_preds.mean(
                dim=1
            )  # [num_nodes_to_query, num_classes]

        # 3) Add noise in batch via Dirichlet mechanism
        self.logger.info("Adding Dirichlet noise to selected nodes (mean teacher predictions)")
        noisy_labels = self.aggregator.batch_add_noise(
            mean_soft_labels
        )  # [num_nodes_to_query, num_classes]

        # Store pseudo labels and pre-noise soft labels
        pseudo_labels_dict = {}
        pre_soft_labels_dict = {}
        for i, node_idx in enumerate(selected_indices):
            node_idx_item = node_idx.item()
            pseudo_labels_dict[node_idx_item] = noisy_labels[i].detach().cpu()
            pre_soft_labels_dict[node_idx_item] = mean_soft_labels[i].detach(
            ).cpu()

        # 4) RDP accounting
        num_queries = len(pseudo_labels_dict)
        if num_queries > 0:
            rdp_result = self.aggregator.compute_total_rdp_cost(num_queries)
            self.epsilon_final = rdp_result["epsilon_final_rdp"]
            self.info["eta"] = self.eta
            self.info["epsilon_final"] = round(self.epsilon_final, 2).item()
            self.info["num_queries"] = self.num_queries

            if "error" in rdp_result:
                self.logger.warning(f"RDP accounting failed: {rdp_result['error']}")
                self.logger.info("RDP privacy cost accounting:")
                self.logger.info(f"  Total queries: {num_queries}")
            else:
                self.logger.info("Privacy cost via RDP accounting (mean teacher predictions):")
                self.logger.info(
                    f"  Total queries: {rdp_result['total_queries']}, eta per query: {rdp_result['eta']:.2f}, "
                    f"Delta_1_F: {rdp_result['Delta_1_F']:.2f}, gamma: {rdp_result['gamma']:.2f}"
                )
                self.logger.info(
                    f"  Final epsilon via RDP (delta={rdp_result['delta']}): {rdp_result['epsilon_final_rdp']:.2f}, "
                    f"best alpha: {rdp_result['best_alpha']:.2f}, "
                    f"valid alphas: {rdp_result['valid_alphas_count']}/{rdp_result['total_alpha_count']}"
                )
                if rdp_result["valid_alphas_count"] < rdp_result["total_alpha_count"]:
                    skipped_count = (
                        rdp_result["total_alpha_count"]
                        - rdp_result["valid_alphas_count"]
                    )
                    self.logger.warning(
                        f"  {skipped_count} alpha orders returned inf (possibly <= alpha * Delta_1_F); skipped"
                    )

                # Save RDP accounting results
                self.rdp_accounting_result = rdp_result
        else:
            self.logger.warning("No pseudo labels generated; cannot run RDP accounting")

        # Pre-noise soft-label distribution
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
            self.info["pre_noise_fraud_leaning_mean"] = fraud_count_pre
            total_pre = len(pre_soft_labels_dict)
            self.logger.info("Pre-noise mean-teacher soft-label distribution:")
            self.logger.info(
                f"  Fraud-leaning: {fraud_count_pre} ({fraud_count_pre / total_pre * 100:.2f}%), "
                f"normal-leaning: {normal_count_pre} ({normal_count_pre / total_pre * 100:.2f}%)"
            )
            if len(fraud_probs_pre) > 0:
                self.logger.info(
                    f"  Fraud-class prob (fraud-leaning) stats: mean={sum(fraud_probs_pre) / len(fraud_probs_pre):.4f}, "
                    f"max={max(fraud_probs_pre):.4f}, min={min(fraud_probs_pre):.4f}"
                )
            if len(normal_probs_pre) > 0:
                self.logger.info(
                    f"  Fraud-class prob (normal-leaning) stats: mean={sum(normal_probs_pre) / len(normal_probs_pre):.4f}, "
                    f"max={max(normal_probs_pre):.4f}, min={min(normal_probs_pre):.4f}"
                )

        # Post-noise pseudo-label distribution
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
            self.info["post_noise_fraud_leaning_mean"] = fraud_count
            self.logger.info("Post-noise pseudo-label distribution (mean teacher predictions):")
            self.logger.info(
                f"  Fraud-leaning: {fraud_count} ({fraud_count / len(pseudo_labels_dict) * 100:.2f}%), "
                f"normal-leaning: {normal_count} ({normal_count / len(pseudo_labels_dict) * 100:.2f}%)"
            )

            if len(fraud_probs) > 0:
                fraud_prob_mean = sum(fraud_probs) / len(fraud_probs)
                fraud_prob_max = max(fraud_probs)
                fraud_prob_min = min(fraud_probs)
                self.logger.info(
                    f"  Fraud-class prob (fraud-leaning) stats: mean={fraud_prob_mean:.4f}, "
                    f"max={fraud_prob_max:.4f}, min={fraud_prob_min:.4f}"
                )

            if len(normal_probs) > 0:
                normal_prob_mean = sum(normal_probs) / len(normal_probs)
                normal_prob_max = max(normal_probs)
                normal_prob_min = min(normal_probs)
                self.logger.info(
                    f"  Fraud-class prob (normal-leaning) stats: mean={normal_prob_mean:.4f}, "
                    f"max={normal_prob_max:.4f}, min={normal_prob_min:.4f}"
                )

        # Store for later use
        self.pre_soft_labels_dict = pre_soft_labels_dict

        # Convert pseudo labels to tensors and store in public_data
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
            f"Pseudo labels saved to public_data (mean teacher predictions); "
            f"{has_pseudo_label.sum().item()} nodes have pseudo labels"
        )

        return pseudo_labels_dict

    def _evaluate_abte_upper_bound(self):
        """
        Generate noise-free pseudo labels on all public nodes with the ABTE
        aggregator and compare with public ground-truth labels to estimate an
        upper-bound performance.
        """

        if self.public_data is None:
            self.logger.warning("public_data is not initialized; cannot evaluate ABTE upper bound")
            return None

        if not hasattr(self, "aggregator") or self.aggregator is None:
            self.logger.warning("aggregator is not initialized; cannot evaluate ABTE upper bound")
            return None

        teacher_preds = getattr(self, "teacher_preds_cache", None)
        node_stats = getattr(self, "node_stats_cache", None)

        if teacher_preds is None or node_stats is None:
            self.logger.warning(
                "Missing teacher prediction cache or node-statistics cache; cannot evaluate ABTE upper bound"
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

        self.logger.info("ABTE upper-bound evaluation (public labels, no privacy noise): %s", metrics)

        pub.abte_upper_bound_logits = agg_logits.detach().cpu()
        pub.abte_upper_bound_probs = agg_probs.detach().cpu()
        self.abte_upper_bound_metrics = metrics

        return metrics
