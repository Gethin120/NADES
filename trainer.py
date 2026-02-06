import torch
from torch import nn
import os
import uuid
from torch.types import Number
from torch.optim import Optimizer
from typing import Iterable, Literal, Optional
from torchmetrics import MeanMetric
from utils.utils import evaluate
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

Stage = Literal["train", "val", "test"]
Metrics = dict[str, Number]


class Trainer:
    def __init__(
        self,
        logger,
        patience: int = 0,
        monitor: str = "val/auc",
        monitor_mode: Literal["min", "max"] = "max",
        val_interval=1,
    ):
        assert monitor_mode in ["min", "max"]

        self.patience = patience  # 训练中断的最大次数
        self.val_interval = val_interval  # 验证间隔
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.logger = logger
        # trainer internal state
        self.model: nn.Module = None
        self.metrics: dict[str, MeanMetric] = {}

    def reset(self):
        self.model = None
        self.metrics = {}

    def update_metrics(
        self, metric_name: str, metric_value: object, weight: int = 1
    ) -> None:
        # if this is a new metric, add it to self.metrics
        device = metric_value.device if torch.is_tensor(metric_value) else "cpu"
        if metric_name not in self.metrics:
            self.metrics[metric_name] = MeanMetric().to(device)

        # update the metric
        self.metrics[metric_name].update(metric_value, weight=weight)

    def aggregate_metrics(self, stage: Stage = "train") -> Metrics:
        metrics = {}

        for metric_name, metric_value in self.metrics.items():
            if stage in metric_name.split("/"):
                value = metric_value.compute()
                metric_value.reset()
                if torch.is_tensor(value):
                    value = value.item()
                metrics[metric_name] = value

        return metrics

    def is_better(self, current_metric: Number, previous_metric: Number) -> bool:
        if self.monitor_mode == "max":
            return current_metric > previous_metric
        elif self.monitor_mode == "min":
            return current_metric < previous_metric
        else:
            raise ValueError(f"Unknown metric mode: {self.monitor_mode}")

    def fit(
        self,
        model: nn.Module,
        epochs: int,
        optimizer: Optimizer,
        global_weights,
        train_dataloader: Iterable,
        val_dataloader: Optional[Iterable] = None,
        test_dataloader: Optional[Iterable] = None,
        checkpoint: bool = False,
        prefix: str = "",
    ) -> Metrics:
        self.model = model
        self.optimizer = optimizer
        self.global_weights = global_weights
        monitor_key = f"{prefix}{self.monitor}"

        if checkpoint:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = os.path.join("checkpoints", f"{uuid.uuid1()}.pt")
            torch.save(self.model.state_dict(), checkpoint_path)

        if val_dataloader is None:
            val_dataloader = []

        if test_dataloader is None:
            test_dataloader = []


        best_metrics = None
        num_epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            metrics = {f"{prefix}epoch": epoch}

            # train loop
            train_metrics = self.loop(train_dataloader, stage="train", prefix=prefix)
            metrics.update(train_metrics)
            # print(metrics)

            # validation loop
            if val_dataloader and self.val_interval and epoch % self.val_interval == 0:
                val_metrics = self.loop(val_dataloader, stage="val", prefix=prefix)
                metrics.update(val_metrics)
                # print(metrics)

                if best_metrics is None or self.is_better(
                    metrics[monitor_key], best_metrics[monitor_key]
                ):
                    best_metrics = metrics
                    num_epochs_without_improvement = 0

                    if checkpoint:
                        torch.save(self.model.state_dict(), checkpoint_path)
                else:
                    num_epochs_without_improvement += 1
                    if num_epochs_without_improvement >= self.patience > 0:
                        break

            # test loop
            if test_dataloader:
                test_metrics = self.loop(test_dataloader, stage="test", prefix=prefix)
                metrics.update(test_metrics)
                # print(metrics)

            # log and update progress
            # Logger.get_instance().log(metrics)
            # self.progress.update(task='epoch', metrics=metrics, advance=1)

        if best_metrics is None:
            best_metrics = metrics
        else:
            # load best model if checkpointing is enabled
            if checkpoint:
                self.model.load_state_dict(torch.load(checkpoint_path))

        # log and return best metrics
        # Logger.get_instance().log_summary(best_metrics)
        self.logger.info(f"Best metrics: {best_metrics}")
        return best_metrics

    def test(self, dataloader: Iterable, prefix: str = "") -> Metrics:
        metrics = self.loop(dataloader, stage="test", prefix=prefix)
        self.logger.info(f"Test metrics: {metrics}")
        return metrics

    def loop(self, dataloader: Iterable, stage: Stage, prefix: str) -> Metrics:
        self.model.train(stage == "train")
        logits_list = []
        y_list = []
        loss_list = []
        count = 0
        for batch in dataloader:
            # metrics = self.step(batch, stage, prefix)
            # for item in metrics:
            #     self.update_metrics(item, metrics[item], weight=len(batch))
            logits, y, loss = self.step(batch, stage, prefix)
            logits_list.append(logits)
            y_list.append(y)
            loss_list.append(loss)
            count += len(batch)
        logits = torch.cat(logits_list, dim=0)
        y = torch.cat(y_list, dim=0).cpu()
        if stage == "test":
            result = evaluate(y, logits)
            # return result
            return {f"{prefix}{stage}/{key}": value for key, value in result.items()}

        else:
            probs = F.softmax(logits, dim=1)[:, 1].cpu().detach().numpy()
            auc = roc_auc_score(y, probs)
            # loss = F.cross_entropy(logits.cpu(), y)
            loss = (
                torch.stack(loss_list).sum() / count if count > 0 else torch.tensor(0.0)
            ).item()
            metrics = {"auc": auc, "loss": loss}
            return {f"{prefix}{stage}/{key}": value for key, value in metrics.items()}


    def step(
        self, batch, stage: Stage, prefix: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stage == "train":
            self.optimizer.zero_grad(set_to_none=True)

        grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(stage == "train")

        loss, logits_masked, y = self.model.step(
            batch, stage=stage, global_weights=self.global_weights
        )

        torch.set_grad_enabled(grad_state)

        if stage == "train" and loss is not None:
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
                self.optimizer.step()

        return logits_masked, y, loss
