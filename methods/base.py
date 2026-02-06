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

        for epoch in range(self.epochs):
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
            loss.backward()

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
