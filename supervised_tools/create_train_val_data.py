from torch.utils.data import DataLoader
from torch_geometric.utils import to_networkx
from utils.data_utils import  to_undirected, Graph_sequence_sampler_pytorch
from torch_geometric.utils import to_dense_adj


def process_subset(subset, max_num_node, max_prev_node):
    '''
    :param subset: dataset for training or dataset for testing
    :param max_num_node: max num of nodes in the set of graphs
    :param max_prev_node: max_num_node - 1
    :return: Graph_sequence_sampler_pytorch data object
    '''

    G_list = []  # list of undirected networkx graphs
    node_attr_list = [] # list of node matrices
    for g in subset:
        node_attr_list.append(g.x)  # node matrix
        nxG = to_networkx(g)
        G_list.append(to_undirected(nxG))

    adj_all = []  # list of A(s) with edge features as elements a_ij [NxNxEf]
    for g in subset:
        adj_all.append(to_dense_adj(edge_index=g.edge_index, batch=None, edge_attr=g.edge_attr))

    return Graph_sequence_sampler_pytorch(Graph_list=G_list, node_attr_list=node_attr_list, adj_all=adj_all,
                                          max_num_node=max_num_node, max_prev_node=max_prev_node)


def create_train_val_dataloaders(trainset, valset, max_num_node, max_prev_node, num_workers=1):
    '''
    for supervised training takes as input:
    - dataset: a list of pyg Data obs,
    - max number of nodes of the loaded graphs,
    - max_prev_node = (max number of nodes-1)
    '''
    train_set = process_subset(trainset, max_num_node, max_prev_node)
    val_set = process_subset(valset, max_num_node, max_prev_node)
    train_dataset_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=num_workers)
    val_dataset_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=num_workers)
    return train_dataset_loader, val_dataset_loader

