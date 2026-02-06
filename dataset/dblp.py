"""
HeteroData(
  author={
    x=[4057, 334],
    y=[4057],
    train_mask=[4057],
    val_mask=[4057],
    test_mask=[4057]
  },
  paper={ x=[14328, 4231] },
  term={ x=[7723, 50] },
  conference={
    num_nodes=20,
    x=[20, 1]
  },
  (author, to, paper)={ edge_index=[2, 19645] },
  (paper, to, author)={ edge_index=[2, 19645] },
  (paper, to, term)={ edge_index=[2, 85810] },
  (paper, to, conference)={ edge_index=[2, 14328] },
  (term, to, paper)={ edge_index=[2, 85810] },
  (conference, to, paper)={ edge_index=[2, 14328] }
)
(['author', 'paper', 'term', 'conference'], [('author', 'to', 'paper'), ('paper', 'to', 'author'), ('paper', 'to', 'term'), ('paper', 'to', 'conference'), ('term', 'to', 'paper'), ('conference', 'to', 'paper')])

"""

import os.path as osp
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
# path = osp.join(osp.dirname(osp.realpath(__file__)), '../../dataset/DBLP')

def dblpdata():
    path = osp.join(osp.realpath("../../"), 'Dataset/DBLP')
    # transform = T.Compose([T.Constant(node_types='conference'), T.RandomNodeSplit('train_rest', num_val=0, num_test=0.2)])

    dataset = DBLP(path, transform=T.Constant(node_types='conference'))  # 为节点增加一个常数特征
    data = dataset[0]
    # print(dataset)
    # print(data.metadata()[0])  # 异质图元数据，借本节点和边
    # print(data.node_types)  # 节点类型
    # print(data.edge_types)  # 边类型
    # print(path)
    # print(data)

    return data