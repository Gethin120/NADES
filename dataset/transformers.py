import torch
import torch.utils.cpp_extension
from torch_geometric.transforms import BaseTransform, RandomNodeSplit
from torch_geometric.data import Data, HeteroData
from torch_sparse import SparseTensor
from torch_geometric.loader.utils import filter_data
from torch_geometric.sampler.utils import to_csc
import torch.nn.functional as F


class BoundOutDegree(BaseTransform):
    def __init__(self, max_out_degree: int):
        self.num_neighbors = max_out_degree
        self.with_replacement = False

    def __call__(self, data: Data) -> Data:
        data.adj_t = data.adj_t.t()
        data = self.sample(data)
        data.adj_t = data.adj_t.t()
        return data

    def sample(self, data: Data) -> Data:
        colptr, row, perm = to_csc(data, device="cpu")
        index = torch.arange(0, data.num_nodes - 1, dtype=int)
        sample_fn = torch.ops.torch_sparse.neighbor_sample
        node, row, col, edge = sample_fn(
            colptr, row, index, [self.num_neighbors], self.with_replacement, True
        )
        data = filter_data(data, node, row, col, edge, perm)
        return data


class BoundDegree(BaseTransform):
    def __init__(self, max_degree: int, detection_type):
        self.max_deg = max_degree
        self.detection_type = detection_type
        try:
            edge_sampler = torch.ops.my_ops.sample_edge
        except (AttributeError, RuntimeError):
            torch.utils.cpp_extension.load(
                name="sampler",
                sources=["csrc/sampler.cpp"],
                build_directory="csrc",
                is_python_module=False,
                verbose=False,
            )
            edge_sampler = torch.ops.my_ops.sample_edge

        self.edge_sampler = edge_sampler

    def __call__(self, data: Data) -> Data:
        N = data[self.detection_type].num_nodes
        edge = (self.detection_type, "homo", self.detection_type)
        E = data[edge].num_edges
        adj: SparseTensor = data[edge].adj_t.t()
        device = adj.device()
        row, col, _ = adj.coo()
        perm = torch.randperm(E)
        row, col = row[perm], col[perm]
        row, col = self.edge_sampler(row.tolist(), col.tolist(), N, self.max_deg)
        adj = SparseTensor(row=row, col=col).to(device)
        data[edge].adj_t = adj.t()
        return data


class BoundDegreePyG(BaseTransform):
    def __init__(self, max_degree: int, detection_type):
        self.max_deg = max_degree
        self.detection_type = detection_type

    def __call__(self, data: Data) -> Data:
        if isinstance(data, HeteroData):
            edge = (self.detection_type, "homo", self.detection_type)
            if hasattr(data[edge], "adj_t"):
                adj = data[edge].adj_t.t()
            else:
                edge_index = data[edge].edge_index
                adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(data[self.detection_type].num_nodes, data[self.detection_type].num_nodes))
            num_nodes = data[self.detection_type].num_nodes
        else:
            if hasattr(data, "adj_t"):
                adj = data.adj_t.t()
            else:
                edge_index = data.edge_index
                adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(data.num_nodes, data.num_nodes))
            num_nodes = data.num_nodes

        device = adj.device()
        row, col, _ = adj.coo()
        row_cpu = row.cpu()
        col_cpu = col.cpu()

        perm = torch.randperm(row_cpu.numel())
        row_cpu = row_cpu[perm]
        col_cpu = col_cpu[perm]

        counts = torch.zeros(num_nodes, dtype=torch.int64)
        sampled_rows = []
        sampled_cols = []
        for r, c in zip(row_cpu.tolist(), col_cpu.tolist()):
            if counts[r] < self.max_deg:
                counts[r] += 1
                sampled_rows.append(r)
                sampled_cols.append(c)

        if sampled_rows:
            row_new = torch.tensor(sampled_rows, dtype=row.dtype, device=device)
            col_new = torch.tensor(sampled_cols, dtype=col.dtype, device=device)
        else:
            row_new = torch.empty((0,), dtype=row.dtype, device=device)
            col_new = torch.empty((0,), dtype=col.dtype, device=device)

        adj_new = SparseTensor(row=row_new, col=col_new, sparse_sizes=adj.sparse_sizes()).to(device)

        if isinstance(data, HeteroData):
            data[edge].adj_t = adj_new.t()
        else:
            data.adj_t = adj_new.t()
        return data


class FilterClassByCount(BaseTransform):
    def __init__(self, min_count: int, remove_unlabeled=False):
        self.min_count = min_count
        self.remove_unlabeled = remove_unlabeled

    def __call__(self, data: Data) -> Data:
        assert hasattr(data, "y")

        y: torch.Tensor = F.one_hot(data.y)
        counts = y.sum(dim=0)
        y = y[:, counts >= self.min_count]
        mask = y.sum(dim=1).bool()  # nodes to keep
        data.y = y.argmax(dim=1)

        if self.remove_unlabeled:
            data = data.subgraph(mask)
        else:
            data.y[~mask] = -1  # set filtered nodes as unlabeled
            if hasattr(data, "train_mask"):
                data.train_mask = data.train_mask & mask
                data.val_mask = data.val_mask & mask
                data.test_mask = data.test_mask & mask

        return data


class GetSubGraph(BaseTransform):
    def __init__(self, num_nodes):
        """
        num_nodes: number of nodes, if float, then it is the percentage of nodes
        """
        self.num_nodes = num_nodes

    def __call__(self, data: Data) -> Data:
        subgraph = RandomNodeSplit(
            split="train_rest", num_val=0, num_test=1 - self.num_nodes
        )(data)

        data = data.subgraph(subgraph.train_mask)
        return data


class GetHomoGraph(BaseTransform):
    """将异质图转换为同质图，只保留(节点，edge，节点）的边,
    edge: 边类型 homo, tt, etc.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data: HeteroData, node_type: str, edge: str) -> Data:
        # 创建一个新的Data对象
        homo_data = Data()

        # 获取第一个节点类型（假设所有节点类型都有相同的特征维度）
        # node_type = data.node_types[0]

        # 复制节点特征和标签
        homo_data.x = data[node_type].x
        if hasattr(data[node_type], "y"):
            homo_data.y = data[node_type].y
        if hasattr(data[node_type], "train_mask"):
            homo_data.train_mask = data[node_type].train_mask
        if hasattr(data[node_type], "val_mask"):
            homo_data.val_mask = data[node_type].val_mask
        if hasattr(data[node_type], "test_mask"):
            homo_data.test_mask = data[node_type].test_mask

        # 只保留同类型的边
        # homo_edges = []
        # for edge_type in data.edge_types:
        #     if edge_type[0] == edge_type[2]:  # 源节点和目标节点类型相同
        #         homo_edges.append(data[edge_type].edge_index)

        # # 合并所有同类型的边
        # if homo_edges:
        #     homo_data.edge_index = torch.cat(homo_edges, dim=1)
        homo_data.edge_index = data[node_type, edge, node_type].edge_index
        return homo_data


class StandardizeFeatures(BaseTransform):
    """
    逐列 Z-Score 标准化 transform。

    对节点特征矩阵的每一列（特征维度）进行 Z-Score 标准化：
    x_normalized = (x - mean) / std

    参数:
        use_train_only: 如果为 True，只使用训练集节点计算均值和标准差（避免数据泄露）
        eps: 防止除零的小值（默认 1e-8）
        attrs: 要标准化的属性列表（默认 ["x"]）
    """

    def __init__(
        self, use_train_only: bool = False, eps: float = 1e-8, attrs: list = None
    ):
        self.use_train_only = use_train_only
        self.eps = eps
        self.attrs = attrs if attrs is not None else ["x"]

    def __call__(self, data: Data) -> Data:
        """
        对节点特征进行逐列 Z-Score 标准化。

        参数:
            data: PyG Data 对象

        返回:
            标准化后的 Data 对象
        """
        for attr in self.attrs:
            if not hasattr(data, attr) or getattr(data, attr) is None:
                continue

            x = getattr(data, attr)

            # 确定用于计算统计量的节点
            if self.use_train_only and hasattr(data, "train_mask"):
                mask = data.train_mask
                if mask.sum() == 0:
                    # 如果没有训练节点，使用所有节点
                    x_stats = x
                else:
                    x_stats = x[mask]
            else:
                x_stats = x

            # 计算每列的均值和标准差
            mean = x_stats.mean(dim=0, keepdim=True)
            std = x_stats.std(dim=0, keepdim=True)

            # Z-Score 标准化: (x - mean) / std
            # 添加 eps 防止除零
            standardized = (x - mean) / (std + self.eps)
            setattr(data, attr, standardized)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(use_train_only={self.use_train_only}, eps={self.eps}, attrs={self.attrs})"
