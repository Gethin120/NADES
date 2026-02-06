import os
import os.path as osp
from typing import Callable, Optional
import numpy as np
from torch_geometric.data import HeteroData, InMemoryDataset, Data
import torch
import pandas as pd


# /home/workspace/Dataset/elliptic++/
class Ellptic(InMemoryDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self):
        return [
            "AddrAddr_edgelist.csv",
            "TxAddr_edgelist.csv",
            "txs_edgelist.csv",
            "wallets_classes.csv",
            " wallets_features_classes_combined.csvAddrTx_edgelist.csv",
            "txs_classes.csv",
            "txs_features.csv",
            "wallets_features.csv",
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def process(self):
        txs_classes = pd.read_csv(osp.join(self.raw_dir, "txs_classes.csv"))
        txs_edges = pd.read_csv(osp.join(self.raw_dir, "txs_edgelist.csv"))
        txs_features = pd.read_csv(osp.join(self.raw_dir, "txs_features.csv"))
        txs_nodes = txs_classes["txId"].values
        txs_map_id = {j: i for i, j in enumerate(txs_nodes)}
        txs_edges.txId1 = txs_edges.txId1.map(txs_map_id)
        txs_edges.txId2 = txs_edges.txId2.map(txs_map_id)
        txs_edges = txs_edges.astype(int)
        txs_edge_index = np.array(txs_edges.values).T  # coo形式的边
        txs_edge_index = torch.tensor(
            txs_edge_index, dtype=torch.long).contiguous()
        txs_labels = torch.tensor(
            txs_classes["class"].values, dtype=torch.long)
        txs_features = torch.tensor(
            np.array(txs_features.drop(["txId", "Time step"], axis=1).values),
            dtype=torch.float32,
        )  # 删除了txId和class列和时间列
        bit_data = HeteroData()
        bit_data["txs"].x = txs_features
        bit_data["txs"].y = txs_labels - 1
        bit_data["txs", "tt", "txs"].edge_index = txs_edge_index
        # wallets数据处理(addr)
        addr_classes = pd.read_csv(
            osp.join(self.raw_dir, "wallets_classes.csv"))
        addr_features = pd.read_csv(
            osp.join(self.raw_dir, "wallets_features.csv"))
        # wallets_features_classes_combined = pd.read_csv(osp.join(path,'wallets_features_classes_combined.csv'))
        addr_features = addr_features.drop(
            ["Time step"], axis=1).drop_duplicates()
        df_merge = addr_classes.merge(
            addr_features, how="left", right_on="address", left_on="address"
        )  # fea存在重复的行，去除后还是用merge比较保险吧
        addr_nodes = addr_classes["address"].values
        addr_map_id = {j: i for i, j in enumerate(addr_nodes)}
        df_merge.address = df_merge.address.map(addr_map_id)
        addr_features = torch.tensor(
            np.array(df_merge.drop(["address", "class"], axis=1).values),
            dtype=torch.float32,
        )
        addr_labels = torch.tensor(df_merge["class"].values, dtype=torch.long)
        # edge数据处理
        AddrAddr_edgelist = pd.read_csv(
            osp.join(self.raw_dir, "AddrAddr_edgelist.csv"))
        AddrAddr_edgelist.input_address = AddrAddr_edgelist.input_address.map(
            addr_map_id
        )
        AddrAddr_edgelist.output_address = AddrAddr_edgelist.output_address.map(
            addr_map_id
        )
        AddrAddr_edgelist = AddrAddr_edgelist.astype(int)
        Addr_edge_index = np.array(AddrAddr_edgelist.values).T  # coo形式的边
        Addr_edge_index = torch.tensor(
            Addr_edge_index, dtype=torch.long).contiguous()

        AddrTx_edgelist = pd.read_csv(
            osp.join(self.raw_dir, "AddrTx_edgelist.csv"))
        AddrTx_edgelist.input_address = AddrTx_edgelist.input_address.map(
            addr_map_id)
        AddrTx_edgelist.txId = AddrTx_edgelist.txId.map(txs_map_id)
        AddrTx_edgelist = AddrTx_edgelist.astype(int)
        AddrTx_edge_index = np.array(AddrTx_edgelist.values).T  # coo形式的边
        AddrTx_edge_index = torch.tensor(
            AddrTx_edge_index, dtype=torch.long
        ).contiguous()

        TxAddr_edgelist = pd.read_csv(
            osp.join(self.raw_dir, "TxAddr_edgelist.csv"))
        TxAddr_edgelist.txId = TxAddr_edgelist.txId.map(txs_map_id)
        TxAddr_edgelist.output_address = TxAddr_edgelist.output_address.map(
            addr_map_id)

        TxAddr_edgelist = TxAddr_edgelist.astype(int)
        TxAddr_edge_index = np.array(TxAddr_edgelist.values).T  # coo形式的边
        TxAddr_edge_index = torch.tensor(
            TxAddr_edge_index, dtype=torch.long
        ).contiguous()
        bit_data["addr"].x = addr_features
        bit_data["addr"].y = addr_labels - 1
        bit_data["addr", "aa", "addr"].edge_index = Addr_edge_index
        bit_data["addr", "at", "txs"].edge_index = AddrTx_edge_index
        bit_data["txs", "ta", "addr"].edge_index = TxAddr_edge_index
        if self.pre_transform is not None:
            bit_data = self.pre_transform(bit_data)

        self.save([bit_data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def preprocess_elliptic_pyg(
    dataset_path="/home/workspace/Dataset/elliptic++/",
    save_path=None,
):
    # ============================================================
    # 0. 设置保存路径（如果未指定，则在dataset_path下）
    # ============================================================
    if save_path is None:
        save_path = osp.join(dataset_path, "elliptic_pyg.pt")

    # ============================================================
    # 1. 检查是否已存在处理后的数据，如果存在则直接加载
    # ============================================================
    if osp.exists(save_path):
        print(f"Loading preprocessed data from {save_path}")
        data = torch.load(save_path)
        return data

    # ============================================================
    # 1. 读取 txs_classes.csv  (txId, class)
    # ============================================================
    df_cls = pd.read_csv(osp.join(dataset_path, "txs_classes.csv"))

    txIds = df_cls["txId"].to_numpy()
    raw_cls = df_cls["class"].to_numpy()

    num_nodes = len(txIds)

    # 建立 txId -> 连续 index 映射
    txid2idx = {int(txid): idx for idx, txid in enumerate(txIds)}

    # y: 1=licit → 0, 2=illicit → 1, 3=unknown → -1
    y = np.full(num_nodes, -1, dtype=np.int64)
    y[raw_cls == 1] = 1
    y[raw_cls == 2] = 0
    # raw_cls == 3 → unknown → y=-1

    marks = y != -1

    # ============================================================
    # 2. 读取特征 txs_features.csv
    # ============================================================
    df_feat = pd.read_csv(osp.join(dataset_path, "txs_features.csv"))

    # 根据你给的列名格式，第一列是 txId，第二列是 Time step，后面都是特征
    assert df_feat.columns[0] == "txId"
    assert df_feat.columns[1] == "Time step"

    feat_txIds = df_feat["txId"].to_numpy()
    time_step = df_feat["Time step"].to_numpy().astype(float)

    # ----------- 重新对齐特征的行顺序到 txs_classes 的顺序 -----------
    # 你的 features.csv 行顺序可能与 classes.csv 不一致，因此必须重排。
    #
    # df_feat 按照 txId → index
    feat_id2row = {int(txid): i for i, txid in enumerate(feat_txIds)}

    # 依照 classes 的 txIds 顺序取 features
    feat_rows = [feat_id2row[int(txid)] for txid in txIds]
    df_feat = df_feat.iloc[feat_rows]
    time_step = time_step[feat_rows]

    # 丢掉 txId 与 Time step，只保留特征部分
    X = df_feat.iloc[:, 2:].to_numpy().astype(float)

    # ============================================================
    # 3. 处理 NaN 特征 (列均值填充)
    # ============================================================
    X = torch.tensor(X, dtype=torch.float32)

    col_mean = torch.nanmean(X, dim=0, keepdim=True)
    col_mean = torch.where(torch.isnan(col_mean),
                           torch.zeros_like(col_mean), col_mean)
    X = torch.where(torch.isnan(X), col_mean, X)

    # ============================================================
    # 4. 标准化 z-score
    # ============================================================
    # mean = X.mean(dim=0, keepdim=True)
    # std = X.std(dim=0, keepdim=True, unbiased=False)
    # std = torch.where(std == 0, torch.ones_like(std), std)
    # X = (X - mean) / std

    # ============================================================
    # 5. 构建 train/val/test mask (基于 Time step)
    # ============================================================
    # 按你预期的划分方式可调整，否则沿用常见 Elliptic 划分规则：
    # train  = t <= 25
    # val    = 26 ~ 34
    # test   = t >= 35
    t = time_step
    private_mask = t <= 38
    public_mask = t > 38

    print(f"Private: {private_mask.sum()}, Public: {public_mask.sum()}")

    # ============================================================
    # 6. 读取边 txs_edgelist.csv，并映射 txId→idx
    # ============================================================
    df_edge = pd.read_csv(osp.join(dataset_path, "txs_edgelist.csv"))

    e_src = df_edge.iloc[:, 0].to_numpy().astype(int)
    e_dst = df_edge.iloc[:, 1].to_numpy().astype(int)

    # 映射成连续 index
    src_idx = np.array([txid2idx[s] for s in e_src], dtype=np.int64)
    dst_idx = np.array([txid2idx[d] for d in e_dst], dtype=np.int64)

    edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)

    # 安全检查，避免 CUDA assert
    assert edge_index.min().item() >= 0
    assert edge_index.max().item() < num_nodes

    # ============================================================
    # 7. 构建 PyG Data
    # ============================================================
    data = Data(
        x=X,
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        private_mask=torch.tensor(private_mask, dtype=torch.bool),
        public_mask=torch.tensor(public_mask, dtype=torch.bool),
        mark=torch.tensor(marks, dtype=torch.bool),
        num_nodes=num_nodes,
    )

    # ============================================================
    # 8. 保存
    # ============================================================
    # 确保目录存在
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)
    print(f"Saved processed graph to {save_path}")

    return data


def elliptic():
    data_base_path = "/home/workspace/Dataset/"
    path = osp.join(data_base_path, "elliptic++/")
    # txs数据处理
    # path = osp.join(osp.realpath("../../"), "Dataset/elliptic++/")
    # path = osp.join(osp.realpath("../../../"), 'Dataset/elliptic++/')# 本地运行是这个path

    elliptic_data = Ellptic(path)
    return elliptic_data


if __name__ == "__main__":
    # elliptic_data = elliptic()
    # data = elliptic_data[0]
    # print(data)
    # homo_data = data.to_homogeneous()
    # print(homo_data)
    data = preprocess_elliptic_pyg()
    print(data)
    print(sum(data.private_mask))
    print(sum(data.public_mask))
    print(sum(data.private_mask) / data.num_nodes)
