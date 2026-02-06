import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
from gaussian_accountant import create_accountant
from matplotlib import pyplot as plt
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    recall_score,
    average_precision_score,
)
from typing import Callable, Dict, List, Optional, Tuple, Union
from torch_geometric.data import Data, Dataset, HeteroData, InMemoryDataset
from torch_geometric.sampler import (
    SamplerOutput,
    NodeSamplerInput,
    HeteroSamplerOutput,
)
from torch_geometric.sampler.utils import remap_keys
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import NodeType
import torch_geometric.typing
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import subgraph
from numpy.typing import ArrayLike, NDArray


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def get_class_weights(data, detection_type: Optional[str] = None):
    # 类别样本数量
    if isinstance(data, Data):
        if hasattr(data, "train_mask"):
            train = data.y[data.train_mask]
        else:
            train = data.y
    elif isinstance(data, HeteroData):
        train = data[detection_type].y[data[detection_type].train_mask]
    else:
        train = data
    train = train[train >= 0]
    class_counts = np.array([sum(train == 0).tolist(), sum(train == 1).tolist()])
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    nonzero = class_counts > 0
    if nonzero.any():
        class_weights[nonzero] = 1.0 / class_counts[nonzero]
        class_weights = class_weights / class_weights.sum()
    else:
        class_weights[:] = 0.5
    return torch.tensor(class_weights, dtype=torch.float32)


# def get_class_weights(data, detection_type: Optional[str]=None):
#     # 类别样本数量
#     if isinstance(data, Data):
#         train = data.y[data.train_mask]
#     else:
#         train = data[detection_type].y[data[detection_type].train_mask]
#
#     class_counts = np.array([np.sum(train == 0), np.sum(train == 1)])
#     class_weights = 1.0 / class_counts  # 权重与样本数量成反比
#
#     # 归一化并处理可能的除零错误
#     class_weights_sum = np.sum(class_weights[class_counts > 0])  # 仅对样本数大于0的类别求和
#     class_weights[class_counts == 0] = 0  # 将样本数为0的类别的权重设为0
#     class_weights[class_counts > 0] /= class_weights_sum  # 对样本数大于0的类别进行归一化
#
#     return torch.tensor(class_weights, dtype=torch.float32)


def clip_and_accumulate(args, model):
    batch_size = args.batch_size
    cum_grads = model.cum_grads
    # C = args.grad_norm_max
    C = 1
    """
    Performs gradient clipping.
    Stores clipped and aggregated gradients.
    """
    # step 0: compute the norms

    # step 1: compute clipping factors

    # step 2: clip
    # step 3: add gaussian noise
    # step 4: assign the new grads, delete the sample grads

    g_norm = Variable(torch.zeros(batch_size), requires_grad=False)
    counter2 = 0
    g_norm = {}
    # print('clip')
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        if p.grad is not None:
            # print(counter2,p.grad.norm(2, dim=-1).shape)
            g_norm[str(counter2)] = p.grad.norm(2, dim=-1)
            # print(counter2,p.grad.shape)
        counter2 += 1
    # print(g_norm)
    # print('accumulate')
    for p, key in zip(
        filter(lambda p: p.requires_grad, model.parameters()), cum_grads.keys()
    ):
        if p.grad is not None and key in g_norm.keys():
            # print(p.grad.shape)
            # print(torch.clamp(g_norm[key].contiguous().view(-1, 1) / C, min=1).shape)
            # print(cum_grads[key].shape)
            cum_grads[key] = F.normalize(
                torch.sum(
                    (
                        p.grad
                        / torch.clamp(g_norm[key].contiguous().view(-1, 1) / C, min=1)
                    ),
                    dim=0,
                ),
                dim=0,
            )
            # print(key,cum_grads[key].shape)
    # print("111")
    # for key, val in cum_grads.items():
    #     print(key,val.shape)


def add_noise(args, model, noise_multiplier):
    cum_grads = model.cum_grads
    C = args.grad_norm_max
    """
    Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
    """
    for p, key in zip(
        filter(lambda p: p.requires_grad, model.parameters()), cum_grads.keys()
    ):
        if p.grad is not None:
            proc_size = p.grad.size(0)
            """
            add noise to summed clipped pars
            compute sigma
            """
            noise = _generate_noise(std=noise_multiplier * C, reference=cum_grads[key])
            noise = F.normalize(noise, dim=0)
            # print(key,proc_size,noise.shape)
            if len(p.data.shape) > 2:
                cum_grads[key] = cum_grads[key].unsqueeze(0).expand(proc_size, -1, -1)
            elif len(p.data.shape) > 1:
                cum_grads[key] = cum_grads[key].expand(proc_size, -1)
            if p.grad.is_cuda:
                p.grad = ((cum_grads[key] + noise).view_as(p.grad)).cuda() / proc_size
            else:
                p.grad = ((cum_grads[key] + noise).view_as(p.grad)) / proc_size


def _generate_noise(
    std: float,
    reference: torch.Tensor,
    generator=None,
) -> torch.Tensor:
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros

    return torch.normal(
        mean=0,
        std=std,
        size=reference.shape,
        device=reference.device,
        generator=generator,
    )


def create_cum_grads(model):
    cum_grads = OrderedDict()
    # print('cum_grads')
    for i, p in enumerate(model.parameters()):
        if p.requires_grad:
            # print(i,p.shape)
            cum_grads[str(i)] = Variable(torch.zeros(p.shape[1:]), requires_grad=False)
    return cum_grads


MAX_SIGMA = 1e6


def get_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: Optional[int] = None,
    steps: Optional[int] = None,
    accountant: str = "rdp",
    epsilon_tolerance: float = 0.01,
    **kwargs,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate

    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        steps: number of steps to run
        accountant: accounting mechanism used to estimate epsilon
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """
    if (steps is None) == (epochs is None):
        raise ValueError(
            "get_noise_multiplier takes as input EITHER a number of steps or a number of epochs"
        )
    if steps is None:
        steps = int(epochs / sample_rate)

    eps_high = float("inf")
    accountant = create_accountant(mechanism=accountant)

    sigma_low, sigma_high = 0, 10
    while eps_high > target_epsilon:
        sigma_high = 2 * sigma_high
        accountant.history = [(sigma_high, sample_rate, steps)]
        eps_high = accountant.get_epsilon(delta=target_delta, **kwargs)
        if sigma_high > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")
    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        accountant.history = [(sigma, sample_rate, steps)]
        eps = accountant.get_epsilon(delta=target_delta, **kwargs)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high


def get_noise():
    def compute_norms(sample_grads):
        batch_size = sample_grads[0].shape[0]
        norms = [
            sample_grad.view(batch_size, -1).norm(2, dim=-1)
            for sample_grad in sample_grads
        ]
        norms = torch.stack(norms, dim=0).norm(2, dim=0)
        return norms, batch_size

    def clip_and_accumulate_and_add_noise(
        model, max_per_sample_grad_norm=1.0, noise_multiplier=1.0
    ):
        sample_grads = tuple(param.grad_sample for param in model.parameters())

        # step 0: compute the norms
        sample_norms, batch_size = compute_norms(sample_grads)

        # step 1: compute clipping factors
        clip_factor = max_per_sample_grad_norm / (sample_norms + 1e-6)
        clip_factor = clip_factor.clamp(max=1.0)

        # step 2: clip
        grads = tuple(
            torch.einsum("i,i...", clip_factor, sample_grad)
            for sample_grad in sample_grads
        )

        # step 3: add gaussian noise
        stddev = max_per_sample_grad_norm * noise_multiplier
        noises = tuple(
            torch.normal(0, stddev, grad_param.shape, device=grad_param.device)
            for grad_param in grads
        )
        grads = tuple(noise + grad_param for noise, grad_param in zip(noises, grads))

        # step 4: assign the new grads, delete the sample grads
        for param, param_grad in zip(model.parameters(), grads):
            param.grad = param_grad / batch_size
            del param.grad_sample


class early_stopper(object):
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initialize the early stopper
        :param patience: the maximum number of rounds tolerated
        :param verbose: whether to stop early
        :param delta: the regularization factor
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_value = None
        self.best_cv = None
        self.is_earlystop = False
        self.count = 0
        self.best_model = None
        # self.val_preds = []
        # self.val_logits = []

    def earlystop(self, loss, model=None):  # , preds, logits):
        """
        :param loss: the loss score on validation set
        :param model: the models
        """
        value = loss
        cv = loss
        # value = ap

        if self.best_value is None:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to("cpu")
            # self.val_preds = preds
            # self.val_logits = logits
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print("EarlyStoper count: {:02d}".format(self.count))
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to("cpu")
            # self.val_preds = preds
            # self.val_logits = logits
            self.count = 0


def plot_loss(avg_train_losses, avg_valid_losses, fold=1):
    # visualize the loss as the network trained
    # fig = plt.figure(figsize=(10, 8))
    plt.plot(
        range(1, len(avg_train_losses) + 1), avg_train_losses, label="Training Loss"
    )
    plt.plot(
        range(1, len(avg_valid_losses) + 1), avg_valid_losses, label="Validation Loss"
    )

    # find position of lowest validation loss
    # minposs = avg_valid_losses.index(min(avg_valid_losses)) + 1
    # plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel("epochs")
    plt.ylabel("loss")
    # plt.ylim(0, 0.5)  # consistent scale
    # plt.xlim(0, len(avg_train_losses) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # name = "loss_plot_" + str(fold) + ".png"
    # fig.savefig(name, bbox_inches='tight')


def evaluate(labels, logits):
    results = {}
    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()
    conf = confusion_matrix(labels, preds)
    recall = recall_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    auc = roc_auc_score(labels, probs)
    gmean = conf_gmean(conf)
    ap = average_precision_score(labels, probs)
    results["auc"] = round(auc, 4)
    results["gmean"] = round(gmean, 4).item()
    results["f1_macro"] = round(f1_macro, 4)
    results["recall"] = round(recall, 4)
    results["ap"] = round(ap, 4)

    return results


def conf_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5


def class_balance(data, detection_type):
    train = data[detection_type].y[data[detection_type].train_mask]
    valid = data[detection_type].y[data[detection_type].val_mask]
    test = data[detection_type].y[data[detection_type].test_mask]
    print(
        f"train: {train.shape[0]}, class 0: {sum(train == 0)}, class 1: {sum(train == 1)}, {sum(train == 1) / train.shape[0]}"
    )
    print(
        f"val: {valid.shape[0]}, class 0: {sum(valid == 0)}, class 1: {sum(valid == 1)}"
    )
    print(
        f"test: {test.shape[0]}, class 0: {sum(test == 0)}, class 1: {sum(test == 1)}"
    )


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        """
        Focal Loss 初始化
        :param alpha: 类别平衡因子，用于调整不同类别的权重
        :param gamma: 对易分类样本的惩罚程度，通常设为 2
        :param reduction: 损失函数的返回方式，可选 'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        计算 Focal Loss
        :param inputs: 模型的预测输出，形状为 [batch_size, num_classes]
        :param targets: 真实标签，形状为 [batch_size]
        :return: Focal Loss 值
        """
        # 将输入的 logits 转换为概率
        probs = F.softmax(inputs, dim=1)

        # 取出目标类别的概率值
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = torch.sum(targets_one_hot * probs, dim=1)

        # 计算 Focal Loss
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-9)

        # 根据 reduction 方式返回损失
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ImbalancedSampler(torch.utils.data.WeightedRandomSampler):
    r"""A weighted random sampler that randomly samples elements according to
    class distribution.
    As such, it will either remove samples from the majority class
    (under-sampling) or add more examples from the minority class
    (over-sampling).

    **Graph-level sampling:**

    .. code-block:: python

        from torch_geometric.loader import DataLoader, ImbalancedSampler

        sampler = ImbalancedSampler(dataset)
        loader = DataLoader(dataset, batch_size=64, sampler=sampler, ...)

    **Node-level sampling:**

    .. code-block:: python

        from torch_geometric.loader import NeighborLoader, ImbalancedSampler

        sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
        loader = NeighborLoader(data, input_nodes=data.train_mask,
                                batch_size=64, num_neighbors=[-1, -1],
                                sampler=sampler, ...)

    You can also pass in the class labels directly as a :class:`torch.Tensor`:

    .. code-block:: python

        from torch_geometric.loader import NeighborLoader, ImbalancedSampler

        sampler = ImbalancedSampler(data.y)
        loader = NeighborLoader(data, input_nodes=data.train_mask,
                                batch_size=64, num_neighbors=[-1, -1],
                                sampler=sampler, ...)

    Args:
        dataset (Dataset or Data or Tensor): The dataset or class distribution
            from which to sample the data, given either as a
            :class:`~torch_geometric.data.Dataset`,
            :class:`~torch_geometric.data.Data`, or :class:`torch.Tensor`
            object.
        input_nodes (Tensor, optional): The indices of nodes that are used by
            the corresponding loader, *e.g.*, by
            :class:`~torch_geometric.loader.NeighborLoader`.
            If set to :obj:`None`, all nodes will be considered.
            This argument should only be set for node-level loaders and does
            not have any effect when operating on a set of graphs as given by
            :class:`~torch_geometric.data.Dataset`. (default: :obj:`None`)
        num_samples (int, optional): The number of samples to draw for a single
            epoch. If set to :obj:`None`, will sample as much elements as there
            exists in the underlying data. (default: :obj:`None`)
    """

    def __init__(
        self,
        dataset: Union[Dataset, Data, List[Data], Tensor],
        input_nodes: Optional[Tensor] = None,
        num_samples: Optional[int] = None,
    ):
        if isinstance(dataset, Data):
            y = dataset.y.view(-1)
            assert dataset.num_nodes == y.numel()
            y = y[input_nodes] if input_nodes is not None else y

        elif isinstance(dataset, Tensor):
            y = dataset.view(-1)
            y = y[input_nodes] if input_nodes is not None else y

        elif isinstance(dataset, InMemoryDataset):
            y = dataset.y.view(-1)
            assert len(dataset) == y.numel()

        else:
            ys = [data.y for data in dataset]
            if isinstance(ys[0], Tensor):
                y = torch.cat(ys, dim=0).view(-1)
            else:
                y = torch.tensor(ys).view(-1)
            assert len(dataset) == y.numel()

        assert y.dtype == torch.long  # Require classification.

        num_samples = y.numel() if num_samples is None else num_samples

        class_weight = 1.0 / y.bincount()
        weight = class_weight[y]
        # negative weight 需要这个权重，return返回的是啥呢？
        return super().__init__(weight, num_samples, replacement=True)

    def sample_from_nodes(
        self, inputs: NodeSamplerInput
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Performs sampling from the nodes specified in :obj:`index`,
        returning a sampled subgraph in the specified output format.

        The :obj:`index` is a tuple holding the following information:

        1. The example indices of the seed nodes
        2. The node indices to start sampling from
        3. The timestamps of the given seed nodes (optional)

        Args:
            index (NodeSamplerInput): The node sampler input object.
            **kwargs (optional): Additional keyword arguments.
        """
        # neg_sampling = NegativeSampling(mode='triplet',amount=6)
        out = self.node_sample(inputs, self._sample)
        if self.subgraph_type == SubgraphType.bidirectional:
            out = out.to_bidirectional()
        return out
        # 输出是一个SamplerOutput

    def node_sample(
        self,
        inputs: NodeSamplerInput,
        sample_fn: Callable,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Performs sampling from a :class:`NodeSamplerInput`, leveraging a
        sampling function that accepts a seed and (optionally) a seed time as
        input. Returns the output of this sampling procedure.
        """
        if inputs.input_type is not None:  # Heterogeneous sampling:
            seed = {inputs.input_type: inputs.node}
            seed_time = None
            if inputs.time is not None:
                seed_time = {inputs.input_type: inputs.time}
        else:  # Homogeneous sampling:
            seed = inputs.node
            seed_time = inputs.time

        out = sample_fn(seed, seed_time)
        out.metadata = (inputs.input_id, inputs.time)

        return out

    def _sample(
        self,
        seed: Union[Tensor, Dict[NodeType, Tensor]],
        seed_time: Optional[Union[Tensor, Dict[NodeType, Tensor]]] = None,
        **kwargs,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Implements neighbor sampling by calling either :obj:`pyg-lib` (if
        installed) or :obj:`torch-sparse` (if installed) sampling routines.
        """
        if isinstance(seed, dict):  # Heterogeneous sampling:
            # TODO Support induced subgraph sampling in `pyg-lib`.
            # if (torch_geometric.typing.WITH_PYG_LIB
            #         and self.subgraph_type != SubgraphType.induced):
            # if torch_geometric.typing.WITH_PYG_LIB:
            #     # TODO (matthias) Ideally, `seed` inherits dtype from `colptr`
            #     colptrs = list(self.colptr_dict.values())
            #     dtype = colptrs[0].dtype if len(colptrs) > 0 else torch.int64
            #     seed = {k: v.to(dtype) for k, v in seed.items()}
            #
            #     args = (
            #         self.node_types,
            #         self.edge_types,
            #         self.colptr_dict,
            #         self.row_dict,
            #         seed,
            #         self.num_neighbors.get_mapped_values(self.edge_types),
            #         self.node_time,
            #     )
            #     if torch_geometric.typing.WITH_EDGE_TIME_NEIGHBOR_SAMPLE:
            #         args += (self.edge_time, )
            #     args += (seed_time, )
            #     if torch_geometric.typing.WITH_WEIGHTED_NEIGHBOR_SAMPLE:
            #         args += (self.edge_weight, )
            #     args += (
            #         True,  # csc
            #         self.replace,
            #         self.subgraph_type != SubgraphType.induced,
            #         self.disjoint,
            #         self.temporal_strategy,
            #         # TODO (matthias) `return_edge_id` if edge features present
            #         True,  # return_edge_id
            #     )
            #
            #     out = torch.ops.pyg.hetero_neighbor_sample(*args)
            #     row, col, node, edge, batch = out[:4] + (None, )
            #
            #     # `pyg-lib>0.1.0` returns sampled number of nodes/edges:
            #     num_sampled_nodes = num_sampled_edges = None
            #     if len(out) >= 6:
            #         num_sampled_nodes, num_sampled_edges = out[4:6]
            #
            #     if self.disjoint:
            #         node = {k: v.t().contiguous() for k, v in node.items()}
            #         batch = {k: v[0] for k, v in node.items()}
            #         node = {k: v[1] for k, v in node.items()}

            if torch_geometric.typing.WITH_TORCH_SPARSE:
                # if self.disjoint:
                #     if self.subgraph_type == SubgraphType.induced:
                #         raise ValueError("'disjoint' sampling not supported "
                #                          "for neighbor sampling with "
                #                          "`subgraph_type='induced'`")
                #     else:
                #         raise ValueError("'disjoint' sampling not supported "
                #                          "for neighbor sampling via "
                #                          "'torch-sparse'. Please install "
                #                          "'pyg-lib' for improved and "
                #                          "optimized sampling routines.")

                out = torch.ops.torch_sparse.hetero_neighbor_sample(
                    self.node_types,
                    self.edge_types,
                    self.colptr_dict,
                    self.row_dict,
                    seed,  # seed_dict
                    self.num_neighbors.get_mapped_values(self.edge_types),
                    self.num_neighbors.num_hops,
                    self.replace,
                    self.subgraph_type != SubgraphType.induced,
                )
                node, row, col, edge, batch = out + (None,)
                num_sampled_nodes = num_sampled_edges = None

            else:
                raise ImportError(
                    f"'{self.__class__.__name__}' requires "
                    f"either 'pyg-lib' or 'torch-sparse'"
                )

            if num_sampled_edges is not None:
                num_sampled_edges = remap_keys(
                    num_sampled_edges,
                    self.to_edge_type,
                )

            return HeteroSamplerOutput(
                node=node,
                row=remap_keys(row, self.to_edge_type),
                col=remap_keys(col, self.to_edge_type),
                edge=remap_keys(edge, self.to_edge_type),
                batch=batch,
                num_sampled_nodes=num_sampled_nodes,
                num_sampled_edges=num_sampled_edges,
            )

        else:  # Homogeneous sampling:
            # TODO Support induced subgraph sampling in `pyg-lib`.
            if (
                torch_geometric.typing.WITH_PYG_LIB
                and self.subgraph_type != SubgraphType.induced
            ):
                args = (
                    self.colptr,
                    self.row,
                    # TODO (matthias) `seed` should inherit dtype from `colptr`
                    seed.to(self.colptr.dtype),
                    self.num_neighbors.get_mapped_values(),
                    self.node_time,
                )
                if torch_geometric.typing.WITH_EDGE_TIME_NEIGHBOR_SAMPLE:
                    args += (self.edge_time,)
                args += (seed_time,)
                if torch_geometric.typing.WITH_WEIGHTED_NEIGHBOR_SAMPLE:
                    args += (self.edge_weight,)
                args += (
                    True,  # csc
                    self.replace,
                    self.subgraph_type != SubgraphType.induced,
                    self.disjoint,
                    self.temporal_strategy,
                    # TODO (matthias) `return_edge_id` if edge features present
                    True,  # return_edge_id
                )

                out = torch.ops.pyg.neighbor_sample(*args)
                row, col, node, edge, batch = out[:4] + (None,)

                # `pyg-lib>0.1.0` returns sampled number of nodes/edges:
                num_sampled_nodes = num_sampled_edges = None
                if len(out) >= 6:
                    num_sampled_nodes, num_sampled_edges = out[4:6]

                if self.disjoint:
                    batch, node = node.t().contiguous()

            elif torch_geometric.typing.WITH_TORCH_SPARSE:
                if self.disjoint:
                    raise ValueError(
                        "'disjoint' sampling not supported for "
                        "neighbor sampling via 'torch-sparse'. "
                        "Please install 'pyg-lib' for improved "
                        "and optimized sampling routines."
                    )

                out = torch.ops.torch_sparse.neighbor_sample(
                    self.colptr,
                    self.row,
                    seed,  # seed
                    self.num_neighbors.get_mapped_values(),
                    self.replace,
                    self.subgraph_type != SubgraphType.induced,
                )
                node, row, col, edge, batch = out + (None,)
                num_sampled_nodes = num_sampled_edges = None

            else:
                raise ImportError(
                    f"'{self.__class__.__name__}' requires "
                    f"either 'pyg-lib' or 'torch-sparse'"
                )

            return SamplerOutput(
                node=node,
                row=row,
                col=col,
                edge=edge,
                batch=batch,
                num_sampled_nodes=num_sampled_nodes,
                num_sampled_edges=num_sampled_edges,
            )


#


def confidence_interval(
    data: ArrayLike,
    func: Callable[[ArrayLike], NDArray] = np.mean,
    size: int = 1000,
    ci: int = 95,
    seed: Optional[int] = None,
) -> float:
    bs_replicates = bootstrap(data, func=func, n_boot=size, seed=seed)
    p = 50 - ci / 2, 50 + ci / 2
    bounds = np.nanpercentile(bs_replicates, p)
    return (bounds[1] - bounds[0]) / 2


def bootstrap(
    data: ArrayLike,
    func: Callable[[ArrayLike], NDArray] = np.mean,
    n_boot: int = 10000,
    seed: Optional[int] = None,
) -> NDArray:
    n = len(data)
    data = np.asarray(data)
    rng = np.random.default_rng(seed)
    integers = rng.integers

    boot_dist = []
    for _ in range(int(n_boot)):
        resampler = integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
        sample = [data.take(resampler, axis=0)]
        boot_dist.append(func(*sample))

    return np.array(boot_dist)


# ----------------------
# Graph splitting utility
# ----------------------


def split_graph_private_public(
    data: Data,
    privacy_ratio: float = 0.7,
    public_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    device: Optional[torch.device] = None,
) -> Tuple[Data, Data]:
    """
    Split a homogeneous graph into a private subgraph and a public subgraph.

    - privacy_ratio controls the node ratio assigned to the private graph (default 0.7)
    - private graph: no train/val/test masks
    - public graph: masks created by torch_geometric.transforms.RandomNodeSplit

    Returns: (private_data, public_data)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert isinstance(data, Data), (
        "Only homogeneous torch_geometric.data.Data is supported"
    )
    assert hasattr(data, "x") and hasattr(data, "y") and hasattr(data, "edge_index")

    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    num_private = int(num_nodes * privacy_ratio)

    private_nodes = perm[:num_private]
    public_nodes = perm[num_private:]

    edge_index = data.edge_index.cpu()

    def build_subgraph(node_idx: torch.Tensor, with_masks: bool) -> Data:
        node_idx = node_idx.cpu()
        node_set = set(node_idx.tolist())
        mask = torch.tensor(
            [
                (int(s.item()) in node_set and int(t.item()) in node_set)
                for s, t in edge_index.t()
            ],
            dtype=torch.bool,
        )
        sub_edges = edge_index[:, mask]
        # remap ids
        node_map = {old: i for i, old in enumerate(node_idx.tolist())}
        remapped = torch.tensor(
            [
                [node_map[int(s.item())], node_map[int(t.item())]]
                for s, t in sub_edges.t()
            ],
            dtype=torch.long,
        ).t()

        d = Data(
            x=data.x[node_idx],
            edge_index=remapped,
            y=data.y[node_idx],
        )

        if with_masks:
            train_r, val_r, test_r = public_split
            assert abs(train_r + val_r + test_r - 1.0) < 1e-6, (
                "public_split must sum to 1.0"
            )
            # RandomNodeSplit expects fractions for val/test; train is implicit remainder
            transform = RandomNodeSplit(
                split="train_rest", num_val=val_r, num_test=test_r, key="y"
            )
            d = transform(d)

        return d.to(device)

    private_data = build_subgraph(private_nodes, with_masks=False)
    public_data = build_subgraph(public_nodes, with_masks=True)

    return private_data, public_data


def split_graph_private_public_pyg(
    data: Data,
    privacy_ratio: float = 0.7,
    public_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    device: Optional[torch.device] = None,
) -> Tuple[Data, Data]:
    """
    Split a homogeneous graph into a private subgraph and a public subgraph.
    This version uses PyTorch Geometric's built-in subgraph utility for better performance.

    - privacy_ratio controls the node ratio assigned to the private graph (default 0.7)
    - private graph: no train/val/test masks
    - public graph: masks created by torch_geometric.transforms.RandomNodeSplit

    Returns: (private_data, public_data)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert isinstance(data, Data), (
        "Only homogeneous torch_geometric.data.Data is supported"
    )
    assert hasattr(data, "x") and hasattr(data, "y") and hasattr(data, "edge_index")

    num_nodes = data.num_nodes

    # Check if public_mask and private_mask exist
    if hasattr(data, "public_mask") and hasattr(data, "private_mask"):
        # Use existing masks to split the graph
        private_mask = data.private_mask
        public_mask = data.public_mask

        # Verify masks are valid
        assert private_mask.dtype == torch.bool and public_mask.dtype == torch.bool, (
            "public_mask and private_mask must be boolean tensors"
        )
        assert len(private_mask) == num_nodes and len(public_mask) == num_nodes, (
            "Mask length must match number of nodes"
        )
        assert (private_mask & public_mask).sum() == 0, (
            "private_mask and public_mask should not overlap"
        )

        # Get node indices from masks
        private_nodes = torch.where(private_mask)[0]
        public_nodes = torch.where(public_mask)[0]
    else:
        # Use random permutation as before
        perm = torch.randperm(num_nodes, device=data.edge_index.device)
        num_private = int(num_nodes * privacy_ratio)
        private_nodes = perm[:num_private]
        public_nodes = perm[num_private:]

    def build_subgraph_pyg(node_idx: torch.Tensor, with_masks: bool) -> Data:
        """Build subgraph using PyG's subgraph utility."""
        # Use PyG's subgraph function to extract edges and remap node IDs
        sub_edge_index, _ = subgraph(
            subset=node_idx,
            edge_index=data.edge_index,
            relabel_nodes=True,  # Automatically remap node IDs to [0, 1, 2, ...]
            num_nodes=num_nodes,
        )

        # Extract node features and labels
        d = Data(
            x=data.x[node_idx],
            edge_index=sub_edge_index,
            y=data.y[node_idx],
        )

        if with_masks:
            train_r, val_r, test_r = public_split
            assert abs(train_r + val_r + test_r - 1.0) < 1e-6, (
                "public_split must sum to 1.0"
            )
            if (d.y < 0).any():
                labeled_mask = d.y >= 0
                if labeled_mask.any():
                    labeled_idx = labeled_mask.nonzero(as_tuple=False).view(-1)
                    train_mask = torch.zeros(
                        d.num_nodes, dtype=torch.bool, device=d.x.device
                    )
                    val_mask = torch.zeros(
                        d.num_nodes, dtype=torch.bool, device=d.x.device
                    )
                    test_mask = torch.zeros(
                        d.num_nodes, dtype=torch.bool, device=d.x.device
                    )

                    gen = torch.Generator(device="cpu")
                    perm = labeled_idx[
                        torch.randperm(labeled_idx.numel(), generator=gen)
                    ].to(d.x.device)

                    n_test = int(round(test_r * labeled_idx.numel()))
                    n_val = int(round(val_r * labeled_idx.numel()))
                    n_test = min(n_test, labeled_idx.numel())
                    n_val = min(n_val, labeled_idx.numel() - n_test)

                    test_idx = perm[:n_test]
                    val_idx = perm[n_test : n_test + n_val]
                    train_idx = perm[n_test + n_val :]

                    if train_idx.numel() == 0 and labeled_idx.numel() > 0:
                        train_idx = perm[:1]
                        remaining = perm[1:]
                        test_idx = remaining[:n_test] if n_test > 0 else test_idx
                        val_idx = (
                            remaining[n_test : n_test + n_val] if n_val > 0 else val_idx
                        )

                    train_mask[train_idx] = True
                    if val_idx.numel() > 0:
                        val_mask[val_idx] = True
                    if test_idx.numel() > 0:
                        test_mask[test_idx] = True

                    d.train_mask = train_mask
                    d.val_mask = val_mask
                    d.test_mask = test_mask
                else:
                    d.train_mask = torch.zeros(
                        d.num_nodes, dtype=torch.bool, device=d.x.device
                    )
                    d.val_mask = torch.zeros(
                        d.num_nodes, dtype=torch.bool, device=d.x.device
                    )
                    d.test_mask = torch.zeros(
                        d.num_nodes, dtype=torch.bool, device=d.x.device
                    )
            else:
                # RandomNodeSplit expects fractions for val/test; train is implicit remainder
                transform = RandomNodeSplit(
                    split="train_rest", num_val=val_r, num_test=test_r, key="y"
                )
                d = transform(d)

        return d.to(device)

    private_data = build_subgraph_pyg(private_nodes, with_masks=False)
    public_data = build_subgraph_pyg(public_nodes, with_masks=True)

    return private_data, public_data
