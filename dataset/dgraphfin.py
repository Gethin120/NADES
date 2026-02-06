import torch
from torch_geometric.data import Dataset
import os.path as osp
import numpy as np
from torch_geometric.data import Data


class DgraphDataset(Dataset):
    def __init__(self):
        super().__init__()


def dgraphfin():
    data_base_path = "/home/workspace/Dataset/"
    path = osp.join(data_base_path, "DGraphFin/dgraphfin.npz")
    dataset = np.load(path)
    print(dataset.files)
    x = torch.tensor(dataset["x"]).float()
    y = torch.tensor(dataset["y"]).int()
    # y = (y == 1).int()
    edge_index = torch.tensor(dataset["edge_index"]).long()
    # Ensure edge_index has shape [2, num_edges] (PyG format)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.t().contiguous()
    # edge_type = dataset["edge_type"]
    # edge_timestamp = dataset["edge_timestamp"]
    train_mask = torch.tensor(dataset["train_mask"]).bool()
    valid_mask = torch.tensor(dataset["valid_mask"]).bool()
    test_mask = torch.tensor(dataset["test_mask"]).bool()
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask
    return data
