import torch
from torch_geometric.utils import to_dense_adj
import numpy as np
import networkx as nx
from args import Args
from utils.chem import numpy_to_rdkit, plot_rdkit_svg_grid, valid_score, numpy_to_smiles, novel_score, unique_score
from rdkit import Chem
from utils.molecular_metrics import MolecularMetrics


def mols_txt(epoch, mols_, smiles_, smiles_list):
    val_score_list = valid_score(mols_, from_numpy=False)  # if valid or not: T/F
    novel_score_list = novel_score(mols_, smiles_list, from_numpy=False)  # if new or not T/F
    unique_score_list = unique_score(mols_, from_numpy=False)

    valid_smiles = []
    for i in range(len(val_score_list)):
        if (val_score_list[i] == True):  # and (novel_score_list[i] == True): # get all valids
            valid_smiles.append(smiles_[i])

    valid_smiles = list(valid_smiles)
    valid_mols = [Chem.MolFromSmiles(smile) for smile in valid_smiles]

    drug_like_val = []
    solub_val = []
    synt_val = []
    for mol in valid_mols:
        try:
            drug_like_val.append(MolecularMetrics.quantitative_estimation_druglikeness_scores([mol]))
        except Exception as e:
            print('Error for druglike', e)
            drug_like_val.append(0)
        try:
            solub_val.append(MolecularMetrics.water_octanol_partition_coefficient_scores([mol], norm=True))
        except Exception as e:
            print('Error for solubility', e)
            solub_val.append(0)
        try:
            synt_val.append(MolecularMetrics.synthetic_accessibility_score_scores([mol], norm=True))
        except Exception as e:
            print('Error for syint', e)
            synt_val.append(0)

    with open('./report/mols_smiles_metrics_epoch_' + str(epoch) + '.txt', 'a') as f:
        f.write(f'GENERATION REPORT for epoch {epoch}:\n')
        f.write('\n')
        f.write(
            f'N of valid mols: {np.sum(val_score_list)}/{len(mols_)}, N of novel mols: {np.sum(novel_score_list)}/{len(mols_)}\n')
        f.write(
            f'Fraction of unique and valid molecules w.r.t. to the number of valid molecules: {unique_score_list}\n')
        f.write('\n')
        f.write(f'Mol #mol_num, SMILE: mol_smile, valid: True or False\n')
        f.write('\n')

        for i in range(len(valid_mols)):
            f.write(
                f'Mol #{i}, SMILE: {str(valid_smiles[i])}, druglikeness_score: {str(drug_like_val[i])}, '
                f'solubility_score: {str(solub_val[i])}, synthesizability_score {str(synt_val[i])}\n')
        f.write('\n')
        f.write(
            f'Average druglikeness_scores {np.mean(np.array(drug_like_val))}, Average solubility {np.mean(np.array(solub_val))},'
            f'Average synthesizability_score {np.mean(np.array(synt_val))}')


def mols_smiles_plots(list_of_pygeom_data, name):
    smiles_ = get_smiles(list_of_pygeom_data)
    indices = range(len(smiles_))

    mols_for_txt = []
    idxs_non_SanitizeMols = []
    for idx in indices:
        m = Chem.MolFromSmiles(smiles_[idx])
        if m:
            mols_for_txt.append(m)
        else:
            idxs_non_SanitizeMols.append(idx)

    smiles_ = [i for j, i in enumerate(smiles_) if j not in idxs_non_SanitizeMols]

    plot_rdkit_svg_grid(mols_for_txt, mols_per_row=5, filename=name)

    return mols_for_txt, smiles_


def get_smiles(dataset):
    smiles_ = []
    for i, obs in enumerate(dataset):

        try:  # this try is mainly used when generating gan dataset

            ef_temp = torch.squeeze(to_dense_adj(edge_index=obs.edge_index, batch=None, edge_attr=obs.edge_attr), 0)

            ef = torch.zeros((ef_temp.shape[0], ef_temp.shape[1], 1))
            adj = torch.zeros((ef_temp.shape[0], ef_temp.shape[1]))

            for row in range(ef_temp.shape[0]):
                for col in range(ef_temp.shape[1]):
                    if int(torch.sum(ef_temp[row, col]).item()) != 0:
                        ef[row, col, 0] = torch.argmax(ef_temp[row, col]).item()
                        adj[row, col] = 1

            ef = np.array(ef)
            adj = np.array(adj)

            adj = adj.astype(int)
            ef = ef.astype(int)

            nf = np.array([torch.argmax(row).item() for row in obs.x])
            nf = np.expand_dims(nf, 1)
            nf = nf.astype(int)

            smiles_.append(numpy_to_smiles(adj, nf, ef))
        except:
            pass

    return smiles_


# ORIGINAL PyG FUNCTION
def to_networkx(data, node_attrs=None, edge_attrs=None):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.DiGraph`.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(data.num_nodes))

    values = {key: data[key].squeeze().tolist() for key in data.keys}

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G


# ORIGINAL NX FUNCTION
def to_undirected(graph):
    """Returns an undirected view of the graph `graph`.

    Identical to graph.to_undirected(as_view=True)
    Note that graph.to_undirected defaults to `as_view=False`
    while this function always provides a view.
    """
    return graph.to_undirected(as_view=True)


def encode_adj(adj, original, max_prev_node, args):
    '''
    :param adj: A of current g with edge features as els : (V, V, 4)
    :param original: plain A of current g (without edge features: a binary matrix)
    :param max_prev_node: number of nodes of current graph - 1
    :return: encoded structure for the edges
    '''

    n = original.shape[0] - 1  # N - 1 of the current graph
    temp = np.zeros((n, max_prev_node, args.edge_feature_dims))

    original_tril = np.tril(original, k=-1)  # lower tri of original A
    original_tril_idx = np.nonzero(original_tril)

    # begin by setting all as absent
    for r in range(temp.shape[0]):
        for c in range(temp.shape[1]):
            temp[r, c, 0] = 1

    for index in range(len(original_tril_idx[0])):
        i = original_tril_idx[0][index]
        j = original_tril_idx[1][index]
        temp[i - 1, j, :] = np.concatenate((np.array([0.]), adj[i, j, :]), 0)
        # [i - 1, j ]  since we drop first row of A in the encoding, we need to 'move up' every row-idx

    adj_output = np.zeros((n, max_prev_node, args.edge_feature_dims))

    # flip
    for i in range(0, n):
        adj_output[i, :i + 1, :] = np.flip(temp[i, :i + 1, :], 0)

    return adj_output


class Graph_sequence_sampler_pytorch(torch.utils.data.Dataset):
    '''
    max_num_node : max number of possible nodes in a graph
    max_prev_node : max previous node that looks back (to lock back at)
    returns : dictionary containing input/output nodes, input/output edges
    '''

    def __init__(self, Graph_list, node_attr_list, adj_all, max_num_node, max_prev_node):
        self.adj_all = adj_all  # list of multidim np.arrays (As) already in edge_feature form [V, V , node_f]
        self.len_all = []  # V for each G
        self.node_attr_list = node_attr_list
        self.graph_list = Graph_list  # list of undirected nx graphs
        for G in Graph_list:
            self.len_all.append(G.number_of_nodes())  # timesteps of node rnn for each G
        self.max_num_node = max_num_node
        self.max_prev_node = max_prev_node
        self.args_ = Args()

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        # edge encoding:
        adj_copy = np.asarray(self.adj_all[idx]).copy()
        adj_copy = np.squeeze(adj_copy)  # adj_copy had bs as first dim
        x_batch = np.zeros((self.max_num_node, self.max_prev_node, self.args_.edge_feature_dims))
        y_batch = np.zeros((self.max_num_node, self.max_prev_node, self.args_.edge_feature_dims))

        original_a = np.asarray(nx.to_numpy_matrix(self.graph_list[idx]))  # A without edge features of the current g
        adj_encoded = encode_adj(adj=adj_copy, original=original_a, max_prev_node=self.max_prev_node, args=self.args_)

        x_batch[0, :, :] = 1
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded

        for r in range(y_batch.shape[0]):
            for c in range(y_batch.shape[1]):
                if np.sum(y_batch[r, c, :]) == 0:
                    y_batch[r, c, 0] = 1

        # node encoding:
        node_attr_list_copy = np.asarray(self.node_attr_list[idx]).copy()
        x_node_attr = np.zeros((self.max_num_node, self.args_.node_feature_dims))
        y_node_attr = np.zeros((self.max_num_node, self.args_.node_feature_dims))

        # input nodes:
        x_node_attr[0, :] = 1
        x_node_attr[1:node_attr_list_copy.shape[0], :] = node_attr_list_copy[:-1]

        # output nodes:
        y_node_attr[:node_attr_list_copy.shape[0], :] = node_attr_list_copy

        len_batch = node_attr_list_copy.shape[0]  # number of nodes of current g

        return {'x': x_batch, 'y': y_batch, 'len': len_batch, 'x_node_attr': x_node_attr, 'y_node_attr': y_node_attr}
