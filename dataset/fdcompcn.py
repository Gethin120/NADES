import os.path as osp
import torch
import dgl
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.io import loadmat
from torch_geometric.utils import from_dgl
from scipy.io import savemat
from dataset.preprocess import get_prr_as_weights, get_weight, visualize_graph, RemoveSelfLoops, get_edge_label
from dataset.transformers import GetSubGraph
from torch_geometric.transforms import RandomNodeSplit, Compose,ToSparseTensor


def compdata():
    data_base_path = '/home/workspace/Dataset/'
    path = osp.join(data_base_path, 'FDCompCN/comp.dgl')
    dataset = dgl.load_graphs(path)[0][0]
    comp_data = from_dgl(dataset)
    # invest_weight = get_prr_as_weights(comp_data['company', 'invest_bc2bc', 'company'].edge_index)
    # provide_weight = get_prr_as_weights(comp_data['company', 'provide_bc2bc', 'company'].edge_index)
    # sale_weight = get_prr_as_weights(comp_data['company', 'sale_bc2bc', 'company'].edge_index)
    # comp_data['company', 'invest_bc2bc', 'company'].edge_weight = torch.tensor(invest_weight.squeeze())
    # comp_data['company', 'provide_bc2bc', 'company'].edge_weight = torch.tensor(provide_weight.squeeze())
    # comp_data['company', 'sale_bc2bc', 'company'].edge_weight = torch.tensor(sale_weight.squeeze())
    comp_data['company'].x = comp_data['company'].feature
    comp_data['company'].y = comp_data['company'].label
    del comp_data['company'].feature
    del comp_data['company'].label
    del comp_data['company'].train_mask
    del comp_data['company'].test_mask
    del comp_data['company'].valid_mask
    del comp_data['company', 'homo', 'company'].label
    del comp_data['company', 'homo', 'company'].train_mask

    # del comp_data['company', 'homo', 'company']

    # path = osp.join(data_base_path, 'FDCompCN/comp_ppr_weitght.mat')
    path = osp.join(data_base_path, 'FDCompCN/comp_weitght_label.mat')
    # path = osp.join(data_base_path, 'FDCompCN/comp_weitght_3.mat')
    comp = loadmat(path)

    # comp_data = HeteroData()

    # node_features = torch.tensor(comp["features"], dtype=torch.float)
    # node_labels = torch.tensor(comp["label"].squeeze()).to(torch.long)
    # comp_data['company'].x = node_features
    # comp_data['company'].y = node_labels

    # invest_bc2bc_tensor = torch.tensor(comp['invest_bc2bc'])
    # provide_bc2bc_tensor = torch.tensor(comp['provide_bc2bc'])
    # sale_bc2bc_tensor = torch.tensor(comp['sale_bc2bc'])
    # comp_data['company', 'invest_bc2bc', 'company'].edge_index = invest_bc2bc_tensor
    # comp_data['company', 'provide_bc2bc', 'company'].edge_index = provide_bc2bc_tensor
    # comp_data['company', 'sale_bc2bc', 'company'].edge_index = sale_bc2bc_tensor

    comp_data['company', 'invest_bc2bc', 'company'].edge_weight = torch.tensor(comp['invest_bc2bc_weight'].squeeze())
    comp_data['company', 'provide_bc2bc', 'company'].edge_weight = torch.tensor(comp['provide_bc2bc_weight'].squeeze())
    comp_data['company', 'sale_bc2bc', 'company'].edge_weight = torch.tensor(comp['sale_bc2bc_weight'].squeeze())
    comp_data['company', 'homo', 'company'].edge_weight = torch.tensor(comp['homo_weight'].squeeze())

    comp_data['company', 'invest_bc2bc', 'company'].edge_label = torch.tensor(comp['invest_bc2bc_label'].squeeze())
    comp_data['company', 'provide_bc2bc', 'company'].edge_label = torch.tensor(comp['provide_bc2bc_label'].squeeze())
    comp_data['company', 'sale_bc2bc', 'company'].edge_label = torch.tensor(comp['sale_bc2bc_label'].squeeze())
    comp_data['company', 'homo', 'company'].edge_label = torch.tensor(comp['homo_label'].squeeze())

    return comp_data


def comp_homo_data()->Data:
    data_base_path = '/home/workspace/Dataset/'
    path = osp.join(data_base_path, 'FDCompCN/comp.dgl')
    dataset = dgl.load_graphs(path)[0][0]
    comp_data = from_dgl(dataset)
    comp_homo = Data()
    comp_homo.x = comp_data['company'].feature
    comp_homo.y = comp_data['company'].label
    comp_homo.edge_index = comp_data['company', 'homo', 'company'].edge_index

    return comp_homo


def compdata_from_dgl(alpha):
    # 从dgl加载，每次计算权重浪费时间，弃用
    data_base_path = '/home/workspace/Dataset/'
    path = osp.join(data_base_path, 'FDCompCN/comp.dgl')
    dataset = dgl.load_graphs(path)[0][0]
    comp_data = from_dgl(dataset)
    comp_data['company'].x = comp_data['company'].feature
    comp_data['company'].y = comp_data['company'].label
    path_save = osp.join(data_base_path, 'FDCompCN/comp_weitght.dgl')
    # invest_weight = get_prr_as_weights(comp_data['company', 'invest_bc2bc', 'company'].edge_index)
    # provide_weight = get_prr_as_weights(comp_data['company', 'provide_bc2bc', 'company'].edge_index)
    # sale_weight = get_prr_as_weights(comp_data['company', 'sale_bc2bc', 'company'].edge_index)
    invest_weight = get_weight(comp_data['company', 'invest_bc2bc', 'company'].edge_index, comp_data['company'].label,
                               alpha=alpha)
    provide_weight = get_weight(comp_data['company', 'provide_bc2bc', 'company'].edge_index, comp_data['company'].label,
                                alpha=alpha)
    sale_weight = get_weight(comp_data['company', 'sale_bc2bc', 'company'].edge_index, comp_data['company'].label,
                             alpha=alpha)
    comp_data['company', 'invest_bc2bc', 'company'].edge_weight = torch.tensor(invest_weight.squeeze())
    comp_data['company', 'provide_bc2bc', 'company'].edge_weight = torch.tensor(provide_weight.squeeze())
    comp_data['company', 'sale_bc2bc', 'company'].edge_weight = torch.tensor(sale_weight.squeeze())
    del comp_data['company'].feature
    del comp_data['company'].label
    del comp_data['company'].train_mask
    del comp_data['company'].test_mask
    del comp_data['company'].valid_mask
    # del comp_data['company', 'homo', 'company']

    return comp_data


def compdata_dgl_2_mat():
    data_base_path = '/home/workspace/Dataset/'
    path = osp.join(data_base_path, 'FDCompCN/comp.dgl')
    dataset = dgl.load_graphs(path)[0][0]
    comp_data = from_dgl(dataset)
    comp = {}
    # comp["features"] = comp_data['company'].feature.numpy()
    # comp['label'] = comp_data['company'].label.numpy()
    # comp['invest_bc2bc'] = comp_data['company', 'invest_bc2bc', 'company'].edge_index.numpy()
    # comp['provide_bc2bc'] = comp_data['company', 'provide_bc2bc', 'company'].edge_index.numpy()
    # comp['sale_bc2bc'] = comp_data['company', 'sale_bc2bc', 'company'].edge_index.numpy()

    # comp['invest_bc2bc_weight'] = get_prr_as_weights(comp_data['company', 'invest_bc2bc', 'company'].edge_index)
    # comp['provide_bc2bc_weight'] = get_prr_as_weights(comp_data['company', 'provide_bc2bc', 'company'].edge_index)
    # comp['sale_bc2bc_weight'] = get_prr_as_weights(comp_data['company', 'sale_bc2bc', 'company'].edge_index)
    # path_save = osp.join(data_base_path, 'FDCompCN/comp_ppr_weitght.mat')

    comp['invest_bc2bc_weight'] = get_prr_as_weights(comp_data['company', 'invest_bc2bc', 'company'].edge_index)
    comp['provide_bc2bc_weight'] = get_prr_as_weights(comp_data['company', 'provide_bc2bc', 'company'].edge_index)
    comp['sale_bc2bc_weight'] = get_prr_as_weights(comp_data['company', 'sale_bc2bc', 'company'].edge_index)
    comp['homo_weight'] = get_prr_as_weights(comp_data['company', 'homo', 'company'].edge_index)

    comp['invest_bc2bc_label'] = get_edge_label(comp_data['company', 'invest_bc2bc', 'company'].edge_index,
                                                comp_data['company'].label)
    comp['provide_bc2bc_label'] = get_edge_label(comp_data['company', 'provide_bc2bc', 'company'].edge_index,
                                                 comp_data['company'].label)
    comp['sale_bc2bc_label'] = get_edge_label(comp_data['company', 'sale_bc2bc', 'company'].edge_index,
                                              comp_data['company'].label)
    comp['homo_label'] = get_edge_label(comp_data['company', 'homo', 'company'].edge_index, comp_data['company'].label)
    path_save = osp.join(data_base_path, 'FDCompCN/comp_weitght_label.mat')
    savemat(path_save, comp)
    print('save mat successfully')


if __name__ == '__main__':
    # compdata_dgl_2_mat()
    # data = compdata_from_dgl(0.9)
    # data = compdata()
    data = comp_homo_data()
    print(data)
    transformer = Compose([GetSubGraph(0.4),
                           RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2, key='y'),
                           ToSparseTensor(remove_edge_index=False)])

    target_data = transformer(data)
    print(target_data)
    # print(data)
    # print(data['company', 'invest_bc2bc', 'company'].edge_index)

    # data = compdata_from_dgl()

    # print(data['company', 'invest_bc2bc', 'company'].edge_index)

    # print(sum(data['company'].y))
    # data_base_path = '/home/workspace/Dataset/'
    # path = osp.join(data_base_path, 'FDCompCN/comp_weitght.mat')
    # comp = loadmat(path)
    # print(comp.keys())
    # from torch_geometric.loader import NeighborLoader,LinkNeighborLoader
    # from torch_geometric.transforms import RandomNodeSplit,RemoveIsolatedNodes,Compose,RemoveDuplicatedEdges
    # from DPHSF.utils import class_balance
    #
    # transform = RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2, key='y')
    # data = transform(data)
    # num_neighbors = {key: [8] for key in data.edge_types}
    # # train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=128,
    # #                               weight_attr='edge_weight', input_nodes=('company', data['company'].train_mask))
    # train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=128,
    #                              input_nodes=('company', data['company'].train_mask))
    # sampled_data = next(iter(train_loader))
    # class_balance(sampled_data,'company')
    # # transform = Compose([RemoveSelfLoops(),RemoveIsolatedNodes(), RemoveDuplicatedEdges()])
    # transform = RemoveSelfLoops()
    # sampled_data =transform(sampled_data)
    # class_balance(sampled_data,'company')
    # visualize_graph(sampled_data,color=sampled_data['company'].y)
    # print('1111')

    '''
    HeteroData(
  company={
    x=[5317, 57],
    y=[5317],
  },
  (company, homo, company)={
    edge_index=[2, 10059],
    label=[2],
    train_mask=[2],
  },
  (company, invest_bc2bc, company)={
    edge_index=[2, 8505],
    edge_weight=[8505],
  },
  (company, provide_bc2bc, company)={
    edge_index=[2, 5944],
    edge_weight=[5944],
  },
  (company, sale_bc2bc, company)={
    edge_index=[2, 6244],
    edge_weight=[6244],
  }
)
    
    '''
