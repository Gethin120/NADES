import os.path as osp
import torch
from scipy.io import loadmat, savemat
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import (
    from_scipy_sparse_matrix,
    degree,
    contains_isolated_nodes,
)
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import from_dgl
from dataset.preprocess import get_prr_as_weights, get_edge_label
from utils.utils import get_class_weights
# from dataset.transformers import GetHomoGraph


def yelpdata():
    data_base_path = "/home/workspace/Dataset/"
    # path = osp.join(data_base_path, 'yelp_a7a80596/YelpChi.mat')
    path = osp.join(data_base_path, "yelp_a7a80596/YelpChi_weight_label.mat")
    yelp = loadmat(path)
    yelp_data = HeteroData()

    node_features = torch.tensor(yelp["features"].todense(), dtype=torch.float)
    node_labels = torch.tensor(yelp["label"].squeeze()).to(torch.long)
    yelp_data["review"].x = node_features
    yelp_data["review"].y = node_labels

    net_rur_tensor, _ = from_scipy_sparse_matrix(yelp["net_rur"])
    net_rtr_tensor, _ = from_scipy_sparse_matrix(yelp["net_rtr"])
    net_rsr_tensor, _ = from_scipy_sparse_matrix(yelp["net_rsr"])
    homo_tensor, _ = from_scipy_sparse_matrix(yelp["homo"])
    yelp_data["review", "net_rur", "review"].edge_index = net_rur_tensor
    yelp_data["review", "net_rtr", "review"].edge_index = net_rtr_tensor
    yelp_data["review", "net_rsr", "review"].edge_index = net_rsr_tensor
    yelp_data["review", "homo", "review"].edge_index = homo_tensor
    # personal PageRank edge weight

    # yelp_data['review', 'net_rur', 'review'].edge_weight = torch.tensor(yelp['net_rur_edge_weight'].squeeze())
    # yelp_data['review', 'net_rtr', 'review'].edge_weight = torch.tensor(yelp['net_rtr_edge_weight'].squeeze())
    # yelp_data['review', 'net_rsr', 'review'].edge_weight = torch.tensor(yelp['net_rsr_edge_weight'].squeeze())
    # yelp_data['review', 'homo', 'review'].edge_weight = torch.tensor(yelp['net_homo_edge_weight'].squeeze())

    # yelp_data['review', 'net_rur', 'review'].edge_label = torch.tensor(yelp['net_rur_edge_label'].squeeze())
    # yelp_data['review', 'net_rtr', 'review'].edge_label = torch.tensor(yelp['net_rtr_edge_label'].squeeze())
    # yelp_data['review', 'net_rsr', 'review'].edge_label = torch.tensor(yelp['net_rsr_edge_label'].squeeze())
    # yelp_data['review', 'homo', 'review'].edge_label = torch.tensor(yelp['net_homo_edge_label'].squeeze())
    return yelp_data


def yelp_cheng_data():
    from dgl import load_graphs

    data_base_path = "/home/workspace/Dataset/"
    path = osp.join(data_base_path, "yelp_a7a80596/graph-yelp.bin")
    graph = load_graphs(path)
    data = from_dgl(graph)
    return data


def add_edges_weight():
    data_base_path = "/home/workspace/Dataset/"
    path = osp.join(data_base_path, "yelp_a7a80596/YelpChi.mat")
    # path = osp.join(data_base_path, 'yelp_a7a80596/YelpChi_ppr_weight.mat')

    yelp = loadmat(path)
    net_rur_tensor, _ = from_scipy_sparse_matrix(yelp["net_rur"])
    net_rtr_tensor, _ = from_scipy_sparse_matrix(yelp["net_rtr"])
    net_rsr_tensor, _ = from_scipy_sparse_matrix(yelp["net_rsr"])
    net_homo_tensor, _ = from_scipy_sparse_matrix(yelp["homo"])
    node_features = torch.tensor(yelp["features"].todense(), dtype=torch.float)
    node_labels = torch.tensor(yelp["label"].squeeze()).to(torch.long)
    yelp_data = HeteroData()
    yelp_data["review"].x = node_features
    yelp_data["review"].y = node_labels
    yelp_data["review", "net_rur", "review"].edge_index = net_rur_tensor
    yelp_data["review", "net_rtr", "review"].edge_index = net_rtr_tensor
    yelp_data["review", "net_rsr", "review"].edge_index = net_rsr_tensor
    yelp_data["review", "homo", "review"].edge_index = net_homo_tensor
    net_rur_edge_weight = get_prr_as_weights(
        yelp_data["review", "net_rur", "review"].edge_index
    )
    net_rtr_edge_weight = get_prr_as_weights(
        yelp_data["review", "net_rtr", "review"].edge_index
    )
    net_rsr_edge_weight = get_prr_as_weights(
        yelp_data["review", "net_rsr", "review"].edge_index
    )
    net_homo_edge_weight = get_prr_as_weights(
        yelp_data["review", "homo", "review"].edge_index
    )

    yelp["net_rur_edge_weight"] = net_rur_edge_weight
    yelp["net_rtr_edge_weight"] = net_rtr_edge_weight
    yelp["net_rsr_edge_weight"] = net_rsr_edge_weight
    yelp["net_homo_edge_weight"] = net_homo_edge_weight

    net_rur_edge_label = get_edge_label(
        yelp_data["review", "net_rur", "review"].edge_index, node_labels
    )
    net_rtr_edge_label = get_edge_label(
        yelp_data["review", "net_rtr", "review"].edge_index, node_labels
    )
    net_rsr_edge_label = get_edge_label(
        yelp_data["review", "net_rsr", "review"].edge_index, node_labels
    )
    net_homo_edge_label = get_edge_label(
        yelp_data["review", "homo", "review"].edge_index, node_labels
    )

    yelp["net_rur_edge_label"] = net_rur_edge_label
    yelp["net_rtr_edge_label"] = net_rtr_edge_label
    yelp["net_rsr_edge_label"] = net_rsr_edge_label
    yelp["net_homo_edge_label"] = net_homo_edge_label
    data_base_path = "/home/workspace/Dataset/"
    path_save = osp.join(data_base_path, "yelp_a7a80596/YelpChi_weight_label.mat")
    savemat(path_save, yelp)
    print("save success")


if __name__ == "__main__":
    # from DPHSF.utils import class_balance
    # add edges weight()
    # add_edges_weight()
    data = yelpdata()
    print(data)
    # transform = GetHomoGraph()
    homo_data = transform(data)
    print(homo_data)
    # homo_data = data.to_homogeneous()
    # print(data)
    # print(homo_data)

    # transform = RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2, key='y')
    # data = transform(data)
    # data_homo = yelp_data_homo()
    # print(data_homo)
    # transform = RandomNodeSplit('random',num_train_per_class=1000, num_val=0.2, num_test=0.2, key='y')
    # transform = RandomNodeSplit('test_rest',num_train_per_class=2670,num_val=0.2, key='y')
    # class_balance(data,'review')
    # from torch_geometric.loader import NeighborLoader,LinkNeighborLoader
    from torch_geometric.sampler import NegativeSampling
    # num_neighbors = {key: [8] * 3 for key in data.edge_types}
    # num_neighbors = [8]
    # num_neighbors = {key: [16] * 1 for key in data.edge_types}
    #
    # train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=1024,
    #                               weight_attr='edge_weight', input_nodes=('review', data['review'].train_mask))
    # train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=1024,
    #                                input_nodes=('review', data['review'].train_mask))
    # sampler = NegativeSampling(mode='triplet',amount=1)
    # sampler = NegativeSampling(mode='binary',amount=0.1)
    # train_loader = LinkNeighborLoader(data,num_neighbors=num_neighbors,batch_size=1024,edge_label_index=data.edge_index,neg_sampling=sampler)
    # sampled_data = next(iter(train_loader))
    #
    # print(sum(data.y))
    # print(sum(sampled_data.y[sampled_data.input_id]))
    # print(sampled_data)
    # print(sum(sampled_data.y))
    # class_balance(sampled_data,'review')
    # degrees = degree(sampled_data['review', 'homo', 'review'].edge_index[1],  dtype=torch.long)
    # print(torch.unique(degrees, return_counts=True))
    # homo_data = sampled_data.to_homogeneous()
    # degrees_homo = degree(homo_data.edge_index[1],  dtype=torch.long)
    # print(torch.unique(degrees_homo, return_counts=True))

    # print(degrees)
    # print(contains_isolated_nodes(sampled_data['review', 'homo', 'review'].edge_index))

    """
 HeteroData(
  review={
    x=[45954, 32],
    y=[45954],
  },
  (review, net_rur, review)={
    edge_index=[2, 98630],
    edge_weight=[98630],
    edge_label=[98630],
  },
  (review, net_rtr, review)={
    edge_index=[2, 1147232],
    edge_weight=[1147232],
    edge_label=[1147232],
  },
  (review, net_rsr, review)={
    edge_index=[2, 6805486],
    edge_weight=[6805486],
    edge_label=[6805486],
  },
  (review, homo, review)={
    edge_index=[2, 7693958],
    edge_weight=[7693958],
    edge_label=[7693958],
  }
)

    
    """
