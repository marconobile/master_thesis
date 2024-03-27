from collections import Counter
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import to_undirected, encode_adj, Graph_sequence_sampler_pytorch
from torch_geometric.utils import to_dense_adj
import torch
import numpy as np
# from networkx import to_numpy_matrix
import networkx as nx
from args import Args

from torch_geometric.utils import to_networkx

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

    data = Graph_sequence_sampler_pytorch(Graph_list=G_list, node_attr_list=node_attr_list, adj_all=adj_all,
                                          max_num_node=max_num_node, max_prev_node=max_prev_node)

    return data


def create_train_val_dataloaders(dataset, max_num_node, max_prev_node):
    '''
    for supervised training takes as input:
    - dataset: a list of pyg Data obs,
    - max number of nodes of the loaded graphs,
    - max_prev_node = (max number of nodes-1)
    '''
    train_set = process_subset(dataset, max_num_node, max_prev_node)
    train_dataset_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    return train_dataset_loader, []


#######################################
### UTILS FOR WEIGHTS COMPUTATIONS: ###
#######################################

def get_log_weights(dataset, args):
    # repeating the pipeline used to pack train data
    # to compute loss weights
    node_feature_dims_ = args.node_feature_dims

    data = process_subset_weights(dataset, max_num_node=args.max_num_node, max_prev_node=args.max_prev_node)
    data_weights = DataLoader(data, batch_size=len(data))

    list_of_label_per_edges = []
    for batch in data_weights:
        for obs in batch['y']:
            for sequence in obs:
                for row in sequence:
                    if torch.sum(row).item() != 0.0:
                        list_of_label_per_edges.append(torch.argmax(row).item())

    list_of_label_per_nodes = []
    for batch in data_weights:
        for obs in batch['y_node_attr']:
            for row in obs:
                if torch.sum(row).item() != 0.0:
                    list_of_label_per_nodes.append(torch.argmax(row).item())

    node_counter = Counter(list_of_label_per_nodes)
    edge_counter = Counter(list_of_label_per_edges)

    node_weights = torch.zeros((node_feature_dims_))
    edge_weights = torch.zeros((args.edge_feature_dims))

    for key, value in node_counter.items():
        node_weights[key] = np.log(sum(node_counter.values()) / (1 + node_counter[key]))
    for key, value in edge_counter.items():
        edge_weights[key] = np.log(sum(edge_counter.values()) / (1 + edge_counter[key]))

    return torch.tensor(node_weights), torch.tensor(edge_weights)


class weights_sampler(Dataset):
    def __init__(self, Graph_list, node_attr_list, adj_all, max_num_node, max_prev_node):
        self.adj_all = adj_all
        self.len_all = []
        self.node_attr_list = node_attr_list
        self.graph_list = Graph_list
        for G in Graph_list:
            self.len_all.append(G.number_of_nodes())
        self.max_num_node = max_num_node
        self.max_prev_node = max_prev_node
        self.args_ = Args()

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj_copy = np.asarray(self.adj_all[idx]).copy()
        adj_copy = np.squeeze(adj_copy)
        x_batch = np.zeros((self.max_num_node, self.max_prev_node, self.args_.edge_feature_dims))
        y_batch = np.zeros((self.max_num_node, self.max_prev_node, self.args_.edge_feature_dims))

        # original_a = np.asarray(to_numpy_matrix(self.graph_list[idx]))
        original_a = nx.adjacency_matrix(self.graph_list[idx]).todense()  # A without edge features of the current g
        adj_encoded = encode_adj(adj=adj_copy, original=original_a, max_prev_node=self.max_prev_node, args=self.args_)

        x_batch[0, :, :] = 1
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded

        node_attr_list_copy = np.asarray(self.node_attr_list[idx]).copy()
        x_node_attr = np.zeros((self.max_num_node, self.args_.node_feature_dims))
        y_node_attr = np.zeros((self.max_num_node, self.args_.node_feature_dims))

        x_node_attr[0, :] = 1
        x_node_attr[1:node_attr_list_copy.shape[0], :] = node_attr_list_copy[:-1]

        y_node_attr[:node_attr_list_copy.shape[0], :] = node_attr_list_copy

        len_batch = node_attr_list_copy.shape[0]

        return {'x': x_batch, 'y': y_batch, 'len': len_batch, 'x_node_attr': x_node_attr, 'y_node_attr': y_node_attr}


def process_subset_weights(subset, max_num_node, max_prev_node):
    G_list = []
    node_attr_list = []
    for g in subset:
        node_attr_list.append(g.x)
        nxG = to_networkx(g)
        G_list.append(to_undirected(nxG))
    adj_all = []
    for g in subset:
        adj_all.append(to_dense_adj(edge_index=g.edge_index, batch=None, edge_attr=g.edge_attr))
    data = weights_sampler(Graph_list=G_list, node_attr_list=node_attr_list, adj_all=adj_all, max_num_node=max_num_node,
                           max_prev_node=max_prev_node)
    return data
