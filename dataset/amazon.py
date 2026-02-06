from scipy.io import savemat
import os.path as osp
import torch
from scipy.io import loadmat
from torch_geometric.data import HeteroData
from torch_geometric.utils import from_scipy_sparse_matrix
from dataset.preprocess import get_prr_as_weights, get_edge_label


def amazondata():
    data_base_path = "/home/workspace/Dataset/"

    path = osp.join(data_base_path, "Amazon/Amazon-fraud.mat")
    # path = osp.join(data_base_path, 'amazon/Amazon-fraud_weight_label.mat')
    amazon = loadmat(path)
    net_upu_tensor, _ = from_scipy_sparse_matrix(amazon["net_upu"])
    net_usu_tensor, _ = from_scipy_sparse_matrix(amazon["net_usu"])
    net_uvu_tensor, _ = from_scipy_sparse_matrix(amazon["net_uvu"])
    net_homo_tensor, _ = from_scipy_sparse_matrix(amazon["homo"])
    node_features = torch.tensor(amazon["features"].todense(), dtype=torch.float)
    node_labels = torch.tensor(amazon["label"].squeeze()).to(torch.long)
    amazon_data = HeteroData()
    amazon_data["user"].x = node_features
    amazon_data["user"].y = node_labels
    amazon_data["user", "net_upu", "user"].edge_index = net_upu_tensor
    amazon_data["user", "net_usu", "user"].edge_index = net_usu_tensor
    amazon_data["user", "net_uvu", "user"].edge_index = net_uvu_tensor
    amazon_data["user", "homo", "user"].edge_index = net_homo_tensor

    # net_upu_weight_tensor = torch.tensor(amazon['net_upu_edge_weight'].squeeze())
    # net_usu_weight_tensor = torch.tensor(amazon['net_usu_edge_weight'].squeeze())
    # net_uvu_weight_tensor = torch.tensor(amazon['net_uvu_edge_weight'].squeeze())
    # net_homo_weight_tensor = torch.tensor(amazon['net_homo_edge_weight'].squeeze())
    #
    # amazon_data['user', 'net_upu', 'user'].edge_weight = net_upu_weight_tensor
    # amazon_data['user', 'net_usu', 'user'].edge_weight = net_usu_weight_tensor
    # amazon_data['user', 'net_uvu', 'user'].edge_weight = net_uvu_weight_tensor
    # amazon_data['user', 'homo', 'user'].edge_weight = net_homo_weight_tensor

    # net_upu_label_tensor = torch.tensor(amazon['net_upu_edge_label'].squeeze())
    # net_usu_label_tensor = torch.tensor(amazon['net_usu_edge_label'].squeeze())
    # net_uvu_label_tensor = torch.tensor(amazon['net_uvu_edge_label'].squeeze())
    # net_homo_label_tensor = torch.tensor(amazon['net_homo_edge_label'].squeeze())

    # amazon_data['user', 'net_upu', 'user'].edge_label = net_upu_label_tensor
    # amazon_data['user', 'net_usu', 'user'].edge_label = net_usu_label_tensor
    # amazon_data['user', 'net_uvu', 'user'].edge_label = net_uvu_label_tensor
    # amazon_data['user', 'homo', 'user'].edge_label = net_homo_label_tensor

    return amazon_data


def add_edges_weight():
    data_base_path = "/home/workspace/Dataset/"
    path = osp.join(data_base_path, "amazon/Amazon-fraud.mat")
    amazon = loadmat(path)
    net_upu_tensor, _ = from_scipy_sparse_matrix(amazon["net_upu"])
    net_usu_tensor, _ = from_scipy_sparse_matrix(amazon["net_usu"])
    net_uvu_tensor, _ = from_scipy_sparse_matrix(amazon["net_uvu"])
    net_homo_tensor, _ = from_scipy_sparse_matrix(amazon["homo"])
    node_features = torch.tensor(amazon["features"].todense(), dtype=torch.float)
    node_labels = torch.tensor(amazon["label"].squeeze()).to(torch.long)
    amazon_data = HeteroData()
    amazon_data["user"].x = node_features
    amazon_data["user"].y = node_labels
    amazon_data["user", "net_upu", "user"].edge_index = net_upu_tensor
    amazon_data["user", "net_usu", "user"].edge_index = net_usu_tensor
    amazon_data["user", "net_uvu", "user"].edge_index = net_uvu_tensor
    amazon_data["user", "homo", "user"].edge_index = net_homo_tensor

    net_upu_edge_weight = get_prr_as_weights(
        amazon_data["user", "net_upu", "user"].edge_index
    )
    net_usu_edge_weight = get_prr_as_weights(
        amazon_data["user", "net_usu", "user"].edge_index
    )
    net_uvu_edge_weight = get_prr_as_weights(
        amazon_data["user", "net_uvu", "user"].edge_index
    )
    net_homo_edge_weight = get_prr_as_weights(
        amazon_data["user", "homo", "user"].edge_index
    )

    amazon["net_upu_edge_weight"] = net_upu_edge_weight
    amazon["net_usu_edge_weight"] = net_usu_edge_weight
    amazon["net_uvu_edge_weight"] = net_uvu_edge_weight
    amazon["net_homo_edge_weight"] = net_homo_edge_weight
    net_upu_edge_label = get_edge_label(
        amazon_data["user", "net_upu", "user"].edge_index, node_labels
    )
    net_usu_edge_label = get_edge_label(
        amazon_data["user", "net_usu", "user"].edge_index, node_labels
    )
    net_uvu_edge_label = get_edge_label(
        amazon_data["user", "net_uvu", "user"].edge_index, node_labels
    )
    net_homo_edge_label = get_edge_label(
        amazon_data["user", "homo", "user"].edge_index, node_labels
    )
    amazon["net_upu_edge_label"] = net_upu_edge_label
    amazon["net_usu_edge_label"] = net_usu_edge_label
    amazon["net_uvu_edge_label"] = net_uvu_edge_label
    amazon["net_homo_edge_label"] = net_homo_edge_label

    data_base_path = "/home/workspace/Dataset/"
    path_save = osp.join(data_base_path, "amazon/Amazon-fraud_weight_label.mat")
    savemat(path_save, amazon)


if __name__ == "__main__":
    # add_edges_weight()
    data = amazondata()
    print(data)
    """
HeteroData(
  user={
    x=[11944, 25],
    y=[11944],
  },
  (user, net_upu, user)={
    edge_index=[2, 351216],
    edge_weight=[351216],
    edge_label=[351216],
  },
  (user, net_usu, user)={
    edge_index=[2, 7132958],
    edge_weight=[7132958],
    edge_label=[7132958],
  },
  (user, net_uvu, user)={
    edge_index=[2, 2073474],
    edge_weight=[2073474],
    edge_label=[2073474],
  },
  (user, homo, user)={
    edge_index=[2, 8796784],
    edge_weight=[8796784],
    edge_label=[8796784],
  }
)

    
    """
