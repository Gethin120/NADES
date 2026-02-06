import torch
from torch_geometric.data import Data
from torch_geometric.transforms import (
    RandomNodeSplit,
    Compose,
    ToSparseTensor,
    NormalizeFeatures,
    NormalizeScale,
    # RemoveIsolatedNodes,
)
from dataset.dgraphfin import dgraphfin
from dataset.amazon import amazondata
from dataset.dblp import dblpdata

# from dataset.preprocess import RemoveSelfLoops
from dataset.yelp import yelpdata

from dataset.fdcompcn import compdata, comp_homo_data
from dataset.elliptic import elliptic, preprocess_elliptic_pyg

from dataset.transformers import FilterClassByCount, StandardizeFeatures
from dataset.transformers import GetHomoGraph


def get_DPSHF_dataset(dataset):
    transform = Compose(
        [
            RandomNodeSplit("train_rest", num_val=0.2, num_test=0.2, key="y"),
            ToSparseTensor(attr=None, remove_edge_index=False),
        ]
    )
    # transformer = RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2, key='y')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset == "amazon":
        data = amazondata()
        data = transform(data)
        data.to(device)
    elif dataset == "dblp":
        data = dblpdata()
        data = transform(data)
        data.to(device)
    elif dataset == "yelp":
        data = yelpdata()
        data = transform(data)
        data.to(device)
    elif dataset == "comp":
        data = compdata()
        data = transform(data)
        data.to(device)

    elif dataset == "elliptic":
        elliptic_data = elliptic()
        data = elliptic_data[0]
        data = transform(data)
        data["txs"].val_mask[data["txs"].y == 2] = False  # 不k折
        data["txs"].train_mask[data["txs"].y == 2] = False
        data["txs"].test_mask[data["txs"].y == 2] = False
    else:
        raise ValueError("Dataset not found")
    return data


def get_NADES_dataset(dataset):
    # transform = Compose(
    #     [
    #         RandomNodeSplit("train_rest", num_val=0.2, num_test=0.2, key="y"),
    #         ToSparseTensor(attr=None, remove_edge_index=False),
    #     ]
    # )
    transform = StandardizeFeatures()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == "amazon":
        detection_type = "user"
        edge = "homo"
        data = amazondata()
        trans = GetHomoGraph()
        data = trans(data, detection_type, edge)
    elif dataset == "yelp":
        detection_type = "review"
        edge = "homo"
        data = yelpdata()
        trans = GetHomoGraph()
        data = trans(data, detection_type, edge)
        # data = transform(data)
        data.to(device)
    elif dataset == "comp":
        detection_type = "company"
        edge = "homo"
        data = compdata()
        trans = GetHomoGraph()
        data = trans(data, detection_type, edge)
        data.to(device)

    elif dataset == "elliptic":
        # detection_type = "txs"
        # edge = "tt"
        # elliptic_data = elliptic()
        # data = elliptic_data[0]
        # trans = GetHomoGraph()
        # data = trans(data, detection_type, edge)
        data = preprocess_elliptic_pyg()
        # data = transform(data)

        data.to(device)

        # data = transform(data)
        # data["txs"].val_mask[data["txs"].y == 2] = False  # 不k折
        # data["txs"].train_mask[data["txs"].y == 2] = False
        # data["txs"].test_mask[data["txs"].y == 2] = False
    elif dataset == "dgraphfin":
        data = dgraphfin()
        # data = transform(data)
        data.to(device)
    else:
        raise ValueError("Dataset not found")
    return data


def get_attack_dataset(dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == "amazon":
        data = amazondata()
        transform = Compose(
            [
                RandomNodeSplit(num_val=0.1, num_test=0.15, key="y"),
                FilterClassByCount(min_count=1000, remove_unlabeled=True),
            ]
        )
        data = transform(data)
        data.to(device)
    # elif dataset == 'dblp':
    #     data = dblpdata()
    #     data = transform(data)
    #     data.to(device)
    elif dataset == "yelp":
        data = yelpdata()
        transform = Compose(
            [
                RandomNodeSplit(num_val=0.1, num_test=0.15, key="y"),
                FilterClassByCount(min_count=1000, remove_unlabeled=True),
            ]
        )
        data = transform(data)
        data.to(device)
    elif dataset == "comp":
        data = comp_homo_data()
        # transform = Compose([
        #     RandomNodeSplit(num_val=0.1, num_test=0.15, key='y'),
        #     FilterClassByCount(min_count=1000, remove_unlabeled=True)])
        # data = transform(data)
        # data.to(device)

    # elif dataset == 'elliptic':
    #     elliptic_data = elliptic()
    #     data = elliptic_data[0]
    #     data = transform(data)
    #     data['txs'].val_mask[data['txs'].y == 2] = False  # 不k折
    #     data['txs'].train_mask[data['txs'].y == 2] = False
    #     data['txs'].test_mask[data['txs'].y == 2] = False
    else:
        raise ValueError("Dataset not found")
    # data = Compose([RemoveSelfLoops(), RemoveIsolatedNodes(), ToSparseTensor()])(data)
    # data = ToSparseTensor(remove_edge_index=False)(data)
    # print_stats(data)
    return data


def print_stats(data: Data):
    nodes_degree: torch.Tensor = data.adj_t.sum(dim=1)
    baseline: float = (
        data.y[data.test_mask].unique(return_counts=True)[1].max().item()
        * 100
        / data.test_mask.sum().item()
    )
    train_ratio: float = data.train_mask.sum().item() / data.num_nodes * 100
    val_ratio: float = data.val_mask.sum().item() / data.num_nodes * 100
    test_ratio: float = data.test_mask.sum().item() / data.num_nodes * 100

    stat = {
        "nodes": f"{data.num_nodes:,}",
        "edges": f"{data.num_edges:,}",
        "features": f"{data.num_features:,}",
        "classes": f"{int(data.y.max() + 1)}",
        "mean degree": f"{nodes_degree.mean():.2f}",
        "median degree": f"{nodes_degree.median()}",
        "train/val/test (%)": f"{train_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f}",
        "baseline acc (%)": f"{baseline:.2f}",
    }
    print(stat)
    # table = dict2table(stat, num_cols=2, title=f'dataset: [yellow]{self.name}[/yellow]')
    # console.info(table)
    # console.print()


if __name__ == "__main__":
    data = get_NADES_dataset("yelp")
    print(data)
