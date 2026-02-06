import torch
from torch import Tensor
from torch_geometric.utils import get_ppr
def get_prr_as_weights(edge_index:Tensor):
    neighbors,weights = get_ppr(edge_index)
    query_tensor = neighbors.t()
    query_weight = weights
    set_tensor = edge_index.t()
    # 初始化set_weight（注意：我们不需要预先用zeros填充它，因为我们会按需更新它）
    ppr_weight = torch.zeros(len(set_tensor), dtype=torch.float)
    # 将set_tensor的每一行转换为元组，并使用这些元组作为字典的键
    # 注意：我们还需要一个列表来存储对应行的索引，以便稍后更新set_weight
    set_dict = {tuple(row): idx for idx, row in enumerate(set_tensor.tolist())}
    # 遍历query_tensor的每一行
    for query_idx, query_row in enumerate(query_tensor.tolist()):
        # 检查当前查询行是否在字典中
        if tuple(query_row) in set_dict:
            # 找到匹配项的索引，并更新set_weight
            match_idx = set_dict[tuple(query_row)]
            ppr_weight[match_idx] = query_weight[query_idx]
    # 注意：此时set_weight只包含找到的匹配项的更新值，其他位置仍然是0
    # 如果你想要未找到匹配项的位置有特定的值（比如保持为0），那么上面的代码已经做到了这一点
    # print(ppr_weight)
    return ppr_weight.numpy()

def get_weight(edge_index,label,alpha=0.7):
    ppr_weight = get_prr_as_weights(edge_index)

    # data.edge_ppr_weight = ppr_weight
    label_row = label[edge_index[0]]
    label_col = label[edge_index[1]]

    edge_weight = torch.exp(-alpha *(label_row != label_col).float())*ppr_weight
    return edge_weight.numpy()
def get_edge_label(edge_index, labels):
    row, col = edge_index
    edge_labels = []
    for i, j in zip(row, col):
        i = i.item()
        j = j.item()
        if labels[i] == labels[j]:
            edge_labels.append(1)
        else:
            edge_labels.append(0)
    return torch.tensor(edge_labels)

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from typing import Optional


def visualize_graph(g, color='red', with_weight: Optional[bool] = False):
    # g是data=dataset[0]
    G = to_networkx(g)

    plt.figure(figsize=(20, 20))
    plt.xticks([])
    plt.yticks([])
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos=pos, with_labels=True,
                     node_color=color, cmap="Set2")

    if with_weight:
        weights = g.edge_weight
        for i, (u, v) in enumerate(G.edges()):
            G[u][v]['weight'] = round(float(weights[i]), 2)
        edge_weights = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_weights)
    plt.show()

from typing import Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_self_loops



class RemoveSelfLoops(BaseTransform):
    r"""Removes all self-loops in the given homogeneous or heterogeneous
    graph (functional name: :obj:`remove_self_loops`).

    Args:
        attr (str, optional): The name of the attribute of edge weights
            or multi-dimensional edge features to pass to
            :meth:`torch_geometric.utils.remove_self_loops`.
            (default: :obj:`"edge_weight"`)
    """
    def __init__(self, attr: str = 'edge_weight') -> None:
        self.attr = attr

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite() or 'edge_index' not in store:
                continue

            store.edge_index, store[self.attr] = remove_self_loops(
                store.edge_index,
                edge_attr=store.get(self.attr, None),
            )

        return data