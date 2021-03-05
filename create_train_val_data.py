from random import shuffle
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from data_utils import to_networkx, to_undirected, encode_adj, Graph_sequence_sampler_pytorch
from torch_geometric.utils import to_dense_adj
import torch
import numpy as np
from networkx import to_numpy_matrix
import torch_geometric
from sklearn.utils.class_weight import compute_class_weight
from args import Args


def process_subset(subset, max_num_node, max_prev_node):
    '''
    :param subset: dataset for training or dataset for testing
        each el of the subset has/is: Data(edge_attr=[E, 4], edge_index=[2, E], x=[V, 4])
    :return: Graph_sequence_sampler_pytorch data object
    '''

    G_list = []  # list of undirected networkx graphs
    node_attr_list = []

    for g in subset:
        node_attr_list.append(g.x)  # node matrix
        nxG = to_networkx(g)
        G_list.append(to_undirected(nxG))

    adj_all = []  # list of A(s) with edge features as elements a_ij nxnxef
    for g in subset:
        adj_all.append(to_dense_adj(edge_index=g.edge_index, batch=None, edge_attr=g.edge_attr))

    data = Graph_sequence_sampler_pytorch(Graph_list=G_list, node_attr_list=node_attr_list, adj_all=adj_all,
                                          max_num_node=max_num_node, max_prev_node=max_prev_node)

    return data


def create_train_val_dataloaders(dataset, max_num_node, max_prev_node):
    '''for supervised training'''
    shuffle(dataset)
    graphs_len = len(dataset)
    graphs_train = dataset[0:int(0.1 * graphs_len)]
    graphs_test = dataset[int(0.90 * graphs_len):]
    train_set = process_subset(graphs_train, max_num_node, max_prev_node)
    test_set = process_subset(graphs_test, max_num_node, max_prev_node)
    train_dataset_loader = DataLoader(train_set, batch_size=32)
    test_dataset_loader = DataLoader(test_set, batch_size=32)

    return train_dataset_loader, test_dataset_loader, graphs_train #graphs_train


def create_train_val_dataloaders_geometric(dataset):
    '''for unsupervised training'''
    shuffle(dataset)
    graphs_len = len(dataset)
    graphs_train = dataset  # [0:int(0.01 * graphs_len)]
    train_dataset_loader = torch_geometric.data.DataLoader(graphs_train, batch_size=32)

    return train_dataset_loader, False  # test_dataset_loader


#######################################
### UTILS FOR WEIGHTS COMPUTATIONS: ###
#######################################

def get_log_weights(dataset,args):

    node_feature_dims_ = args.node_feature_dims

    data = process_subset_weights(dataset, max_num_node=100, max_prev_node=99)
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
    edge_weights = torch.zeros((5))

    for key, value in node_counter.items():
        node_weights[key] = np.log(sum(node_counter.values()) / (1 + node_counter[key]))
    for key, value in edge_counter.items():
        edge_weights[key] = np.log(sum(edge_counter.values()) / (1 + edge_counter[key]))

    return torch.tensor(node_weights), torch.tensor(edge_weights)


# def get_ratio_weights(dataset):
#     data = process_subset_weights(dataset)
#     data_weights = DataLoader(data, batch_size=len(data))
#
#     list_of_label_per_edges = []
#     for batch in data_weights:
#         for obs in batch['y']:
#             for sequence in obs:
#                 for row in sequence:
#                     if torch.sum(row).item() != 0.0:
#                         list_of_label_per_edges.append(torch.argmax(row).item())
#
#     list_of_label_per_nodes = []
#     for batch in data_weights:
#         for obs in batch['y_node_attr']:
#             for row in obs:
#                 if torch.sum(row).item() != 0.0:
#                     list_of_label_per_nodes.append(torch.argmax(row).item())
#
#     node_counter = Counter(list_of_label_per_nodes)
#     edge_counter = Counter(list_of_label_per_edges)
#
#     node_weights = torch.ones(4) / torch.tensor(list(node_counter.values()))
#     edge_weights = torch.ones(5) / torch.tensor(list(edge_counter.values()))
#
#     return torch.tensor(node_weights), torch.tensor(edge_weights)


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
        x_batch = np.zeros((self.max_num_node, self.max_prev_node, 5))
        y_batch = np.zeros((self.max_num_node, self.max_prev_node, 5))

        original_a = np.asarray(to_numpy_matrix(self.graph_list[idx]))
        adj_encoded = encode_adj(adj=adj_copy, original=original_a, max_prev_node=self.max_prev_node)

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


def get_sklearn_weights(dataset, max_num_node, max_prev_node):
    data = process_subset_weights(dataset, max_num_node, max_prev_node)
    data_weights = DataLoader(data, batch_size=len(dataset))

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

    edge_weights = compute_class_weight('balanced', np.unique(list_of_label_per_edges), list_of_label_per_edges)
    node_weights = compute_class_weight('balanced', np.unique(list_of_label_per_nodes), list_of_label_per_nodes)

    return torch.tensor(node_weights), torch.tensor(edge_weights)
