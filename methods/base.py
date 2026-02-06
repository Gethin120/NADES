import torch
import numpy as np
from abc import ABC, abstractmethod
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from dataset.loader import NodeDataLoader
from trainer import Trainer
from utils.utils import plot_loss, evaluate, get_class_weights
from typing import Optional, Any, List
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from modules.base import TrainableModule, Metrics, Stage
import copy


class MethodBase(ABC):
    @abstractmethod
    def fit(self, data: Data, prefix: str = "") -> Metrics:
        """Fit the model to the given data."""

    @abstractmethod
    def test(self, data: Optional[Data] = None, prefix: str = "") -> Metrics:
        """Test the model on the given data, or the training data if data is None."""

    @abstractmethod
    def predict(self, data: Optional[Data] = None) -> Tensor:
        """Predict the labels for the given data, or the training data if data is None."""


class FraudDetection(MethodBase):
    def __init__(
        self,
        detection_type,
        epochs,
        num_neighbor,
        num_layer,
        batch_size,
        lr,
        logger,
        patience,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_type = detection_type
        self.epochs = epochs
        self.num_layer = num_layer
        self.num_neighbor = num_neighbor
        self.batch_size = batch_size
        self.data = None  # data is kept for caching purposes
        self.lr = lr
        self.logger = logger
        self.patience = patience
        # self.trainer = Trainer(logger=self.logger, patience=50)
        self.trainer = Trainer(
            logger=self.logger,
            patience=self.patience,
            monitor="val/auc",
            monitor_mode="max",
        )

    def dataset_process(self, data):
        return data

    def data_loader(self, data: Data, stage):
        """Return a dataloader for the given stage."""

        batch_size = self.batch_size
        dataloader = NodeDataLoader(
            data=data,
            stage=stage,
            batch_size=batch_size,
            # batch_size='full',
            # shuffle=(stage == 'train'),
            shuffle=True,
            drop_last=False,
            poisson_sampling=False,
        )

        return dataloader

    @property
    @abstractmethod
    def detector(self) -> Module:
        """Return the underlying classifier."""

    def reset_parameters(self):
        self.data = None

    def pre_fit(self, data):
        """把数据转换成homo"""
        if isinstance(data, Data):
            data_new = data
        else:
            edge = (self.detection_type, "homo", self.detection_type)
            data_new = Data(
                x=data[self.detection_type].x,
                y=data[self.detection_type].y,
                edge_index=data[edge].edge_index,
                adj_t=data[edge].adj_t,
                train_mask=data[self.detection_type].train_mask,
                val_mask=data[self.detection_type].val_mask,
                test_mask=data[self.detection_type].test_mask,
            )

        self.data = data_new.to(self.device, non_blocking=True)

    def fit(self, data, prefix: str = ""):
        """Fit the model to the given data."""
        self.data = data
        metrics = self._train(self.data, prefix=prefix)
        # print(metrics)
        # self.logger.info(f'Train metrics: {metrics}')
        # test_metrics = self.test(self.data, prefix=prefix)
        # print(test_metrics)

        # self._train_mini_batch(self.data)
        # self._train_all(self.data)
        # train_metrics = self._train(self.data)
        # test_metrics = self.test(self.data, prefix=prefix)
        # return {**train_metrics, **test_metrics}
        return metrics

    def test(self, data: Optional[Data] = None, prefix: str = "") -> Metrics:
        """Predict the labels for the given data, or the training data if data is None."""
        if data is None:
            data = self.data

        data = data.to(self.device, non_blocking=True)

        test_metics = self.trainer.test(
            dataloader=self.data_loader(data, "test"),
            prefix=prefix,
        )

        return test_metics

    def predict(self, data: Optional[Data] = None) -> Tensor:
        """Predict the labels for the given data, or the training data if data is None."""
        if data is None:
            data = self.data

        data = data.to(self.device, non_blocking=True)
        return self.detector.predict(data)

    def _train(self, data: Data, prefix: str = "") -> Metrics:
        self.logger.info("training fraud detection")
        self.detector.to(self.device)
        self.global_weights = get_class_weights(data, self.detection_type).to(
            self.device
        )

        metrics = self.trainer.fit(
            model=self.detector,
            epochs=self.epochs,
            optimizer=self.configure_optimizer(),
            train_dataloader=self.data_loader(data, "train"),
            val_dataloader=self.data_loader(data, "val"),
            test_dataloader=self.data_loader(data, "test"),
            checkpoint=False,
            global_weights=self.global_weights,
            prefix=prefix,
        )

        return metrics

    def _train_mini_batch(self, data: Data):
        self.detector.to(self.device)
        train_loader = self.data_loader(data, "train")
        val_loader = self.data_loader(data, "val")
        test_loader = self.data_loader(data, "test")
        optimizer = self.configure_optimizer()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=30, verbose=True
        )

        global_weights = get_class_weights(data, self.detection_type).to(self.device)
        avg_train_losses = []
        avg_valid_losses = []
        valid_probs = torch.zeros(
            [data[self.detection_type].num_nodes], dtype=torch.float
        ).to(self.device)
        test_predictions = torch.zeros(
            [data[self.detection_type].num_nodes, 2], dtype=torch.float
        ).to(self.device)
        best_valid_auc = 0.0
        best_epoch = 0
        result = {}
        self.detector.train()
        for epoch in range(self.epochs):
            train_losses = []
            for step, data_batch in enumerate(train_loader):
                data_batch = data_batch.to(self.device)
                batch_size = data_batch[self.detection_type].batch_size
                optimizer.zero_grad()
                # x = data_batch[self.detection_type].x
                # edge = (self.detection_type, 'homo', self.detection_type)
                # edge_index = data_batch[edge].edge_index
                out = self.detector(data)[:batch_size]
                class_weights = 0.9 * global_weights + 0.1 * get_class_weights(
                    data_batch, self.detection_type
                ).to(self.device)
                loss = F.cross_entropy(
                    out,
                    data_batch[self.detection_type].y[:batch_size],
                    weight=class_weights,
                )
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                probs = F.softmax(out, dim=1)[:, 1].cpu().detach().numpy()
                auc = roc_auc_score(
                    data_batch[self.detection_type].y[:batch_size].cpu(), probs
                )
                print(
                    "In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, "
                    "train_auc:{:.4f}".format(epoch, step, loss, auc)
                )
            train_loss = np.average(train_losses)
            avg_train_losses.append(train_loss)
            # val_loss_list = 0
            # val_all_list = 0
            self.detector.eval()
            with torch.no_grad():
                valid_losses = []
                for step, data_batch in enumerate(val_loader):
                    data_batch = data_batch.to(self.device)
                    batch_size = data_batch[self.detection_type].batch_size
                    out = self.detector(data_batch)[:batch_size]
                    loss = F.cross_entropy(
                        out, data_batch[self.detection_type].y[:batch_size]
                    )
                    # val_loss_list = val_loss_list + loss
                    probs = F.softmax(out, dim=1)[:, 1]
                    auc = roc_auc_score(
                        data_batch[self.detection_type].y[:batch_size].cpu(),
                        probs.cpu().detach().numpy(),
                    )
                    # val_all_list += batch_size
                    valid_losses.append(loss.item())
                    # print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, '
                    #       'val_auc:{:.4f}'.format(epoch, step, loss, auc))
                    val_batch_id = data_batch[self.detection_type].input_id
                    valid_probs[val_batch_id] = probs
                valid_loss = np.average(valid_losses)
                avg_valid_losses.append(valid_loss)
                scheduler.step(valid_loss)
                val_mask = data[self.detection_type].val_mask
                label = data[self.detection_type].y[val_mask].cpu()  # 真实标签
                valid_auc = roc_auc_score(
                    label, valid_probs[val_mask].cpu().detach().numpy()
                )
                # print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
                if best_valid_auc < valid_auc:
                    best_valid_auc = valid_auc
                    best_epoch = epoch
                    with torch.no_grad():
                        for step, data_batch in enumerate(test_loader):
                            data_batch = data_batch.to(self.device)
                            batch_size = data_batch[self.detection_type].batch_size
                            out = self.detector(data_batch)[:batch_size]
                            test_batch_id = data_batch[self.detection_type].input_id
                            test_predictions[test_batch_id] = out
                            # if step % 10 == 0:
                            #     print('In test batch:{:04d}'.format(step))
                    test_mask = data[self.detection_type].test_mask
                    # 真实标签
                    label = data[self.detection_type].y[test_mask].cpu()
                    result = evaluate(label, test_predictions[test_mask])

            epoch_len = len(str(self.epochs))
            print_msg = (
                f"[{epoch + 1:>{epoch_len}}/{self.epochs:>{epoch_len}}] "
                + f"train_loss: {train_loss:.5f} "
                + f"valid_loss: {valid_loss:.5f}"
            )
            print(print_msg)
        plot_loss(avg_train_losses, avg_valid_losses)
        if self.logger is not None:
            self.logger.info(
                f"In Epoch{best_epoch}, AUC: {result['auc']:.4f}, G-mean: {result['gmean']:.4f}, F1-macro: {result['f1_macro']:.4f}, Recall: {result['recall']:.4f}, AP: {result['ap']:.4f}"
            )

    def _train_all(self, data: Data):
        optimizer = self.configure_optimizer()
        self.detector.to(self.device)

        train_mask = data[self.detection_type].train_mask
        val_mask = data[self.detection_type].val_mask
        test_mask = data[self.detection_type].test_mask
        train_losses = []
        valid_losses = []
        best_valid_auc = 0.0
        result = {}
        best_epoch = 0
        global_weights = get_class_weights(data, self.detection_type).to(self.device)
        # assert check_per_sample_gradients_are_correct(
        #     data[self.detection_type].x,
        #     self.detector
        # )
        for epoch in range(self.epochs):
            # if epoch % 100 == 0:
            #     print(torch.cuda.memory_summary(device=torch.device('cuda')))#输出显存使用情况
            self.detector.train()
            optimizer.zero_grad()
            logits = self.detector(data)
            train_logits = logits[train_mask]
            loss = F.cross_entropy(
                train_logits,
                data[self.detection_type].y[train_mask],
                weight=global_weights,
            )
            train_losses.append(loss.item())
            # print(self.detector.hesp.forward.__code__.co_varnames)
            loss.backward()

            # for name, param in self.detector.named_parameters():
            #     print(3333333333)
            #     if isinstance(param, torch.nn.parameter.UninitializedParameter):
            #         print(f"Parameter {name} is uninitialized.")
            #     if param.grad is None:
            #         print(f"{name} has no gradient!")
            optimizer.step()
            train_probs = F.softmax(train_logits, dim=1)[:, 1].cpu().detach().numpy()
            train_auc = roc_auc_score(
                data[self.detection_type].y[train_mask].cpu(), train_probs
            )
            print(
                "In epoch:{:03d}, train_loss:{:4f}, train_auc:{:.4f}".format(
                    epoch, loss, train_auc
                )
            )
            del train_logits, train_probs
            self.detector.eval()
            with torch.no_grad():
                # logits = self.detector(data)  # 是否重新计算logits呢？
                val_logits = logits[val_mask]
                valid_loss = F.cross_entropy(
                    val_logits, data[self.detection_type].y[val_mask]
                )
                valid_probs = F.softmax(val_logits, dim=1)[:, 1].cpu().detach().numpy()
                valid_auc = roc_auc_score(
                    data[self.detection_type].y[val_mask].cpu(), valid_probs
                )
                valid_losses.append(valid_loss.item())
                print(
                    "In epoch:{:03d}, valid_loss:{:4f}, valid_auc:{:.4f}".format(
                        epoch, valid_loss, valid_auc
                    )
                )

                if best_valid_auc < valid_auc:
                    best_valid_auc = valid_auc
                    test_logits = logits[test_mask]
                    result = evaluate(
                        data[self.detection_type].y[test_mask].cpu(), test_logits
                    )
                    best_epoch = epoch
                del val_logits, valid_probs, logits
        plot_loss(train_losses, valid_losses)

        # print(
        #     f"In Epoch{best_epoch}, AUC: {result['auc']:.4f}, G-mean: {result['gmean']:.4f}, F1-macro: {result['f1_macro']:.4f}, Recall: {result['recall']:.4f}, AP: {result['ap']:.4f}")
        if self.logger is not None:
            self.logger.info(
                f"In Epoch{best_epoch}, AUC: {result['auc']:.4f}, G-mean: {result['gmean']:.4f}, F1-macro: {result['f1_macro']:.4f}, Recall: {result['recall']:.4f}, AP: {result['ap']:.4f}"
            )

    def configure_trainer(self, **kwargs) -> Trainer:
        trainer = Trainer(monitor="val/auc", monitor_mode="max", **kwargs)
        return trainer

    def configure_optimizer(self) -> Optimizer:
        optimizer = torch.optim.Adam(
            self.detector.parameters(), lr=self.lr, weight_decay=1e-4
        )

        return optimizer


class FederatedFraudDetection(FraudDetection):
    """
    联邦版本的欺诈检测基类。

    该类实现了基本的联邦学习框架，包括：
    - 数据分区到多个客户端
    - 联邦训练循环（客户端选择、本地训练、模型聚合）
    - 基于FedAvg的模型聚合算法

    子类需要实现：
    - _create_detector()方法：创建检测器模型实例

    子类可以重写（可选）：
    - _partition_data()方法：实现数据分区逻辑（默认使用简单随机节点分区）
    - _local_train()方法：实现客户端本地训练逻辑（默认使用标准训练器）
    - _aggregate_models()方法：实现模型聚合逻辑（默认使用FedAvg加权平均）
    """

    def __init__(
        self,
        detection_type: str,
        epochs: int,
        num_neighbor: int,
        num_layer: int,
        batch_size: int,
        lr: float,
        logger: Any,
        patience: int,
        # 联邦学习特定参数
        num_clients: int = 5,
        global_rounds: int = 50,
        local_epochs: int = 5,
        client_fraction: float = 1.0,
        **kwargs,
    ):
        """
        初始化联邦欺诈检测器。

        Args:
            detection_type: 检测类型
            epochs: 训练轮数（用于本地训练）
            num_neighbor: 邻居数量
            num_layer: 层数
            batch_size: 批次大小
            lr: 学习率
            logger: 日志记录器
            patience: 早停耐心值
            num_clients: 客户端数量
            global_rounds: 全局聚合轮数
            local_epochs: 每轮本地训练轮数
            client_fraction: 每轮参与的客户端比例
            **kwargs: 其他参数
        """
        super().__init__(
            detection_type=detection_type,
            epochs=epochs,
            num_neighbor=num_neighbor,
            num_layer=num_layer,
            batch_size=batch_size,
            lr=lr,
            logger=logger,
            patience=patience,
        )

        # 联邦学习参数
        self.num_clients = num_clients
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.client_fraction = client_fraction

        # 联邦学习状态
        self.global_model: Optional[Module] = None
        self.client_data: List[Data] = []
        self.client_models: List[Module] = []
        self._initial_detector: Optional[Module] = None

    @property
    def detector(self) -> Module:
        """
        返回检测器模型。

        如果全局模型已初始化，返回全局模型；否则返回初始检测器。
        子类必须实现 _create_detector() 方法来创建初始检测器。
        """
        if self.global_model is not None:
            return self.global_model
        if self._initial_detector is None:
            self._initial_detector = self._create_detector()
        return self._initial_detector

    def _create_detector(self) -> Module:
        """
        创建初始检测器模型。

        子类必须实现此方法以创建检测器模型。

        Returns:
            检测器模型实例
        """
        raise NotImplementedError(
            "Subclass must implement _create_detector() method to create the detector model"
        )

    def fit(self, data: Data, prefix: str = "") -> Metrics:
        """
        执行联邦训练。

        Args:
            data: 训练数据
            prefix: 指标前缀

        Returns:
            训练指标
        """
        self.pre_fit(data)
        self.logger.info("=" * 60)
        self.logger.info("Federated Fraud Detection Training")
        self.logger.info("=" * 60)

        # 初始化全局模型
        if self.global_model is None:
            # 使用detector属性创建全局模型（会调用_create_detector如果尚未创建）
            self.global_model = copy.deepcopy(self.detector)

        # 数据分区
        self.logger.info(f"Partitioning data into {self.num_clients} clients")
        self.client_data = self._partition_data(self.data)
        self.logger.info(f"Created {len(self.client_data)} client partitions")

        # 初始化客户端模型
        self.logger.info("Initializing client models")
        self.client_models = [
            copy.deepcopy(self.global_model) for _ in range(len(self.client_data))
        ]

        # 联邦训练
        self.logger.info(f"Starting federated training: {self.global_rounds} rounds")
        metrics = self._federated_training(prefix=prefix)

        self.logger.info("=" * 60)
        self.logger.info("Federated Training Completed")
        self.logger.info("=" * 60)

        return metrics

    def _partition_data(self, data: Data) -> List[Data]:
        """
        将图数据分区到多个客户端。

        默认实现：简单的随机节点分区（子类可以重写以实现更复杂的分区策略）。

        Args:
            data: 输入图数据

        Returns:
            客户端数据列表
        """
        self.logger.info(
            f"Partitioning data into {self.num_clients} clients using random node split"
        )

        num_nodes = data.num_nodes
        nodes_per_client = num_nodes // self.num_clients

        client_data_list = []
        for i in range(self.num_clients):
            start_idx = i * nodes_per_client
            end_idx = (
                (i + 1) * nodes_per_client if i < self.num_clients - 1 else num_nodes
            )

            # 创建节点掩码
            node_mask = torch.zeros(num_nodes, dtype=torch.bool)
            node_mask[start_idx:end_idx] = True

            # 提取子图（简化版本，实际应该考虑边）
            client_data = Data(
                x=data.x[node_mask],
                y=data.y[node_mask] if hasattr(data, "y") else None,
                edge_index=None,  # 简化版本，实际需要提取子图的边
                train_mask=data.train_mask[node_mask]
                if hasattr(data, "train_mask")
                else None,
                val_mask=data.val_mask[node_mask]
                if hasattr(data, "val_mask")
                else None,
                test_mask=data.test_mask[node_mask]
                if hasattr(data, "test_mask")
                else None,
            )
            client_data = client_data.to(self.device)
            client_data_list.append(client_data)

            self.logger.info(
                f"Client {i + 1}: {client_data.num_nodes} nodes, "
                f"{client_data.train_mask.sum().item() if hasattr(client_data, 'train_mask') and client_data.train_mask is not None else 0} train nodes"
            )

        return client_data_list

    def _federated_training(self, prefix: str = "") -> Metrics:
        """
        执行联邦训练循环。

        Args:
            prefix: 指标前缀

        Returns:
            最佳训练指标
        """
        best_val_auc = 0.0
        best_round = 0
        best_metrics = None

        for round_idx in range(self.global_rounds):
            self.logger.info(
                f"\n--- Global Round {round_idx + 1}/{self.global_rounds} ---"
            )

            # 选择客户端
            num_selected = max(1, int(len(self.client_data) * self.client_fraction))
            selected_clients = np.random.choice(
                len(self.client_data), size=num_selected, replace=False
            )
            self.logger.info(f"Selected {num_selected} clients: {selected_clients}")

            # 客户端本地训练
            client_weights = []
            client_sizes = []

            for client_idx in selected_clients:
                client_data = self.client_data[client_idx]
                client_model = self.client_models[client_idx]

                # 将全局模型参数复制到客户端
                client_model.load_state_dict(self.global_model.state_dict())

                # 本地训练
                self._local_train(
                    client_model=client_model,
                    client_data=client_data,
                    client_idx=client_idx,
                    round_idx=round_idx,
                )

                # 收集模型参数和数据大小用于聚合
                client_weights.append(client_model.state_dict())
                client_sizes.append(
                    client_data.train_mask.sum().item()
                    if hasattr(client_data, "train_mask")
                    and client_data.train_mask is not None
                    else client_data.num_nodes
                )

            # 模型聚合
            aggregated_state = self._aggregate_models(client_weights, client_sizes)
            self.global_model.load_state_dict(aggregated_state)

            # 评估全局模型
            val_auc = self._evaluate_global_model(stage="val")
            self.logger.info(f"Round {round_idx + 1}: Val AUC = {val_auc:.4f}")

            # 跟踪最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_round = round_idx + 1
                best_metrics = {
                    f"{prefix}val/auc": val_auc,
                    f"{prefix}best_round": best_round,
                }

        # 最终评估
        test_auc = self._evaluate_global_model(stage="test")
        self.logger.info(f"Final Test AUC = {test_auc:.4f}")

        if best_metrics is None:
            best_metrics = {f"{prefix}val/auc": 0.0, f"{prefix}test/auc": test_auc}
        else:
            best_metrics[f"{prefix}test/auc"] = test_auc

        return best_metrics

    def _local_train(
        self,
        client_model: Module,
        client_data: Data,
        client_idx: int,
        round_idx: int,
    ):
        """
        客户端本地训练。

        子类可以重写此方法以实现自定义的本地训练逻辑。

        Args:
            client_model: 客户端模型
            client_data: 客户端数据
            client_idx: 客户端索引
            round_idx: 全局轮次索引
        """
        optimizer = torch.optim.Adam(client_model.parameters(), lr=self.lr)
        global_weights = get_class_weights(client_data, self.detection_type).to(
            self.device
        )

        # 创建训练器
        trainer = Trainer(
            logger=self.logger,
            patience=self.local_epochs,
            monitor="val/loss",
            monitor_mode="min",
        )

        # 训练客户端模型
        trainer.fit(
            model=client_model,
            epochs=self.local_epochs,
            optimizer=optimizer,
            train_dataloader=self.data_loader(client_data, "train"),
            val_dataloader=self.data_loader(client_data, "val"),
            test_dataloader=None,
            checkpoint=False,
            global_weights=global_weights,
            prefix=f"client_{client_idx + 1}/",
        )

    def _aggregate_models(
        self, client_weights: List[dict], client_sizes: List[int]
    ) -> dict:
        """
        聚合客户端模型参数（FedAvg算法）。

        子类可以重写此方法以实现自定义的聚合算法。

        Args:
            client_weights: 客户端模型状态字典列表
            client_sizes: 客户端数据大小列表

        Returns:
            聚合后的模型状态字典
        """
        total_size = sum(client_sizes)
        if total_size == 0:
            raise ValueError("Total client size is zero")

        # 初始化聚合参数
        aggregated_state = {}
        for key in client_weights[0].keys():
            aggregated_state[key] = torch.zeros_like(client_weights[0][key])

        # FedAvg: 加权平均
        for client_state, size in zip(client_weights, client_sizes):
            weight = size / total_size
            for key in aggregated_state.keys():
                aggregated_state[key] += weight * client_state[key]

        self.logger.debug(
            f"Aggregated {len(client_weights)} client models using FedAvg "
            f"(total samples: {total_size})"
        )

        return aggregated_state

    def _evaluate_global_model(self, stage: str = "val") -> float:
        """
        评估全局模型在验证集或测试集上的性能。

        Args:
            stage: 评估阶段（"val" 或 "test"）

        Returns:
            AUC分数
        """
        self.global_model.eval()

        all_logits = []
        all_labels = []

        # 在每个客户端的数据上评估
        for client_data in self.client_data:
            client_data = client_data.to(self.device)

            # 获取相应的掩码
            if stage == "val":
                if (
                    hasattr(client_data, "val_mask")
                    and client_data.val_mask is not None
                ):
                    mask = client_data.val_mask
                else:
                    continue
            elif stage == "test":
                if (
                    hasattr(client_data, "test_mask")
                    and client_data.test_mask is not None
                ):
                    mask = client_data.test_mask
                else:
                    continue
            else:
                raise ValueError(f"Unknown stage: {stage}")

            if not mask.any():
                continue

            with torch.no_grad():
                # 简化版本：假设detector可以直接处理数据
                # 实际实现可能需要根据具体模型调整
                if hasattr(self.global_model, "__call__"):
                    logits = self.global_model(client_data)
                else:
                    # 如果模型需要edge_index等，需要从原始数据中提取
                    logits = self.global_model(client_data.x, client_data.edge_index)

                logits_masked = logits[mask]
                y_masked = client_data.y[mask]

                # 过滤未标记节点
                label_mask = y_masked != -1
                if label_mask.any():
                    logits_labeled = logits_masked[label_mask]
                    y_labeled = y_masked[label_mask]
                    all_logits.append(logits_labeled.cpu())
                    all_labels.append(y_labeled.cpu())

        # 聚合所有客户端的结果
        if len(all_logits) == 0:
            return 0.0

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 计算AUC
        probs = F.softmax(all_logits, dim=1)[:, 1].cpu().detach().numpy()
        try:
            auc = roc_auc_score(all_labels.numpy(), probs)
        except ValueError:
            auc = 0.0

        return auc

    def predict(self, data: Optional[Data] = None) -> Tensor:
        """
        使用全局模型进行预测。

        Args:
            data: 预测数据，如果为None则使用训练数据

        Returns:
            预测结果
        """
        if self.global_model is None:
            raise ValueError("Global model not initialized. Please call fit() first.")

        if data is None:
            if self.data is None:
                raise ValueError("No data provided and no cached data available")
            data = self.data

        data = data.to(self.device)
        self.global_model.eval()

        with torch.no_grad():
            if hasattr(self.global_model, "__call__"):
                logits = self.global_model(data)
            else:
                logits = self.global_model(data.x, data.edge_index)
            return torch.softmax(logits, dim=-1)


class NodeClassification(MethodBase):
    """用于节点分类任务的基类。
    该类实现了节点分类的基本功能，包括训练、测试和预测。
    子类需要实现classifier属性方法来提供具体的分类模型。
    """

    def __init__(
        self,
        epochs: int,
        lr: float,
        batch_size: int,
        full_batch_eval: bool,
        device: torch.device,
        logger: Any,
    ):
        """初始化节点分类器。

        Args:
            num_classes: 分类的类别数量
            epochs: 训练的轮数
            lr: 学习率
            batch_size: 批处理大小
            full_batch_eval: 是否在评估阶段使用完整批次
            device: 训练设备
            logger: 日志记录器
            weight_decay: 权重衰减系数，默认为1e-4
        """
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = 1e-4
        self.batch_size = batch_size
        self.full_batch_eval = full_batch_eval
        self.logger = logger
        self.device = device
        # 缓存训练数据
        self.data = None

        # 初始化训练器
        self.trainer = Trainer(logger=self.logger, patience=50)

    @property
    @abstractmethod
    def classifier(self) -> TrainableModule:
        """返回底层分类器模型。

        Returns:
            用于节点分类的模型
        """

    def reset_parameters(self) -> None:
        """重置模型参数和训练状态。"""
        self.classifier.reset_parameters()
        self.trainer.reset()
        self.data = None

    def fit(self, data: Data, prefix: str = "") -> Metrics:
        """训练模型。

        Args:
            data: 训练数据
            prefix: 指标前缀

        Returns:
            训练和测试指标的字典
        """
        self.data = data.to(self.device, non_blocking=True)
        train_metrics = self._train(self.data, prefix=prefix)
        return train_metrics

    def test(self, data: Optional[Data] = None, prefix: str = "") -> Metrics:
        """测试模型性能。

        Args:
            data: 测试数据，如果为None则使用训练数据
            prefix: 指标前缀

        Returns:
            测试指标的字典
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data provided and no cached data available")
            data = self.data

        data = data.to(self.device, non_blocking=True)

        test_metrics = self.trainer.test(
            dataloader=self.data_loader(data, "test"),
            prefix=prefix,
        )

        return test_metrics

    def predict(self, data: Optional[Data] = None) -> Tensor:
        """预测节点类别。

        Args:
            data: 预测数据，如果为None则使用训练数据

        Returns:
            节点类别预测结果
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data provided and no cached data available")
            data = self.data

        data = data.to(self.device, non_blocking=True)
        return self.classifier.predict(data)

    def _train(self, data: Data, prefix: str = "") -> Metrics:
        """训练模型的内部方法。

        Args:
            data: 训练数据
            prefix: 指标前缀

        Returns:
            训练指标的字典
        """
        self.logger.info("training attacker")
        self.classifier.to(self.device)

        metrics = self.trainer.fit(
            model=self.classifier,
            epochs=self.epochs,
            optimizer=self.configure_optimizer(),
            train_dataloader=self.data_loader(data, "train"),
            val_dataloader=self.data_loader(data, "val"),
            test_dataloader=self.data_loader(data, "test"),
            global_weights=None,
            checkpoint=False,
            prefix=prefix,
        )

        return metrics

    def configure_optimizer(self) -> Optimizer:
        """配置优化器。

        Returns:
            配置好的优化器实例
        """
        return torch.optim.Adam(
            self.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def data_loader(self, data: Data, stage: Stage) -> NodeDataLoader:
        """为不同阶段创建数据加载器。

        Args:
            data: 数据
            stage: 阶段('train', 'val'或'test')

        Returns:
            配置好的数据加载器
        """
        batch_size = self.batch_size
        dataloader = NodeDataLoader(
            data=data,
            stage=stage,
            batch_size=batch_size,
            # batch_size='full',
            # shuffle=(stage == 'train'),
            shuffle=True,
            drop_last=False,
            poisson_sampling=False,
        )

        return dataloader


class LinkPrediction(MethodBase):
    def __init__(
        self,
        epochs,
        lr,
        batch_size: int,
        full_batch_eval: bool,
        device: torch.device,
        logger,
    ):
        self.data = None
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = 1e-4
        self.batch_size = batch_size
        self.device = device
        self.logger = logger
        self.trainer = Trainer(logger=self.logger, patience=50)

    @property
    @abstractmethod
    def link_predictor(self) -> TrainableModule:
        """Return the underlying classifier."""

    def reset_parameters(self):
        self.link_predictor.reset_parameters()
        self.trainer.reset()

    def fit(
        self,
        train_data: Data,
        val_data: Optional[Data] = None,
        test_data: Optional[Data] = None,
        prefix: str = "",
    ) -> Metrics:
        """Fit the model to the given data."""
        # train_data = train_data.to(self.device, non_blocking=True)
        if val_data is not None:
            val_data = val_data.to(self.device, non_blocking=True)
        if test_data is not None:
            test_data = test_data.to(self.device, non_blocking=True)

        train_metrics = self._train(train_data, val_data, test_data, prefix=prefix)
        test_metrics = (
            self.test(test_data, prefix=prefix) if test_data is not None else {}
        )
        return {**train_metrics, **test_metrics}

    def test(self, data: Optional[Data] = None, prefix: str = "") -> Metrics:
        """Predict the labels for the given data, or the training data if data is None."""

        data = data.to(self.device, non_blocking=True)

        test_metics = self.trainer.test(
            dataloader=[data],
            prefix=prefix,
        )

        return test_metics

    def predict(self, data: Optional[Data] = None) -> Tensor:
        """Predict the labels for the given data, or the training data if data is None."""
        data = data.to(self.device, non_blocking=True)
        return self.link_predictor.predict(data)

    def _train(
        self,
        train_data: Data,
        val_data: Optional[Data] = None,
        test_data: Optional[Data] = None,
        prefix: str = "",
    ) -> Metrics:
        self.logger.info("training link predictor")
        self.link_predictor.to(self.device)

        metrics = self.trainer.fit(
            model=self.link_predictor,
            epochs=self.epochs,
            optimizer=self.configure_optimizer(),
            train_dataloader=[train_data],
            val_dataloader=[val_data] if val_data is not None else None,
            test_dataloader=[test_data]
            if test_data is not None and globals["debug"]
            else None,
            checkpoint=True,
            prefix=prefix,
            global_weights=None,
        )

        return metrics

    def configure_optimizer(self) -> Optimizer:
        """配置优化器。

        Returns:
            配置好的优化器实例
        """
        return torch.optim.Adam(
            self.link_predictor.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def data_loader(self, data: Data, stage: Stage) -> NodeDataLoader:
        """为不同阶段创建数据加载器。

        Args:
            data: 数据
            stage: 阶段('train', 'val'或'test')

        Returns:
            配置好的数据加载器
        """
        batch_size = self.batch_size
        dataloader = NodeDataLoader(
            data=data,
            stage=stage,
            batch_size=batch_size,
            # batch_size='full',
            # shuffle=(stage == 'train'),
            shuffle=True,
            drop_last=False,
            poisson_sampling=False,
        )

        return dataloader
