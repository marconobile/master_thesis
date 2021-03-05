import torch
from chem import numpy_to_rdkit
import numpy as np
from torch_geometric.utils import to_dense_adj
import networkx as nx
from data_utils import to_networkx, to_undirected
from molecular_metrics import MolecularMetrics
from rdkit import Chem


# def retrieve_idx_fail_sanitized(clean_dataset):
#     # this needs clean_dataset
#     mols_ = []
#     indices_to_drop = []
#     try:
#         for i, obs in enumerate(clean_dataset):
#
#             ef_temp = torch.squeeze(to_dense_adj(edge_index=obs.edge_index, batch=None, edge_attr=obs.edge_attr), 0)
#
#             ef = torch.zeros((ef_temp.shape[0], ef_temp.shape[1], 1))
#             adj = torch.zeros((ef_temp.shape[0], ef_temp.shape[1]))
#
#             for row in range(ef_temp.shape[0]):
#                 for col in range(ef_temp.shape[1]):
#                     if int(torch.sum(ef_temp[row, col]).item()) != 0:
#                         ef[row, col, 0] = torch.argmax(ef_temp[row, col]).item() + 1
#                         adj[row, col] = 1
#
#             ef = np.array(ef)
#             adj = np.array(adj)
#
#             adj = adj.astype(int)
#             ef = ef.astype(int)
#
#             nf = np.array([torch.argmax(row).item() for row in obs.x])
#             nf = np.expand_dims(nf, 1)
#             nf = nf.astype(int)
#             try:
#                 mols_.append(numpy_to_rdkit(adj, nf, ef, sanitize=True))
#             except:
#                 # print('HOLA 1')
#                 indices_to_drop.append(i)
#     except:
#         # print('HOLA 2')
#         return [], []
#
#     return mols_, indices_to_drop


def drop_non_sanitizable(dataset, indices_to_drop):
    idxs_to_be_dropped = set(indices_to_drop)
    len_ = len(dataset)

    dataset_ = []
    for idx in range(len_):
        if idx not in idxs_to_be_dropped:
            dataset_.append(dataset[idx])

    return dataset_


def set_reward(dataset, rdkit_mols, reward_type):
    if reward_type == 'druglikeness':
        y_arr = MolecularMetrics.quantitative_estimation_druglikeness_scores(rdkit_mols)
    elif reward_type == 'solubility':
        y_arr = MolecularMetrics.water_octanol_partition_coefficient_scores(rdkit_mols)
    elif reward_type == 'synthesizability':
        y_arr = MolecularMetrics.synthetic_accessibility_score_scores(rdkit_mols)
    elif reward_type == 'joint':
        druglikeness = MolecularMetrics.quantitative_estimation_druglikeness_scores(rdkit_mols)
        solubility = MolecularMetrics.water_octanol_partition_coefficient_scores(rdkit_mols, norm=True)
        synthesizability = MolecularMetrics.synthetic_accessibility_score_scores(rdkit_mols, norm=True)
        # np.array([m, n, j])  # row 1=m, row 2=n row 3=j
        y_arr = np.array([druglikeness, solubility, synthesizability])
    elif reward_type == 'valid':
        y_arr_list = []
        for mol in rdkit_mols:
            try:
                Chem.SanitizeMol(mol)
                y_arr_list.append(1)
            except:
                y_arr_list.append(0)
        y_arr = np.array(y_arr_list)

    y_arr = torch.from_numpy(y_arr)
    y_arr.to(dtype=torch.float32)

    if reward_type == 'druglikeness' or reward_type == 'synthesizability' or reward_type == 'solubility' or reward_type == 'valid':
        for i, obs in enumerate(dataset):
            obs.y = torch.unsqueeze(y_arr[i],0)#y_arr[i]
    elif reward_type == 'joint':
        for i, obs in enumerate(dataset):
            obs.y = y_arr[:, i]

    return dataset


def set_y_single_obs(mol, reward_type):
    if reward_type == "valid":
        try:
            Chem.SanitizeMol(mol[0])
            y_arr = np.ones(1)
            # return True
        except ValueError:
            # return False
            y_arr = np.zeros(1)

    if reward_type == 'druglikeness':
        y_arr = MolecularMetrics.quantitative_estimation_druglikeness_scores(mol)
    elif reward_type == 'solubility':
        y_arr = MolecularMetrics.water_octanol_partition_coefficient_scores(mol)
    elif reward_type == 'synthesizability':
        y_arr = MolecularMetrics.synthetic_accessibility_score_scores(mol)
    elif reward_type == 'joint':
        druglikeness = MolecularMetrics.quantitative_estimation_druglikeness_scores(mol)
        solubility = MolecularMetrics.water_octanol_partition_coefficient_scores(mol, norm=True)
        synthesizability = MolecularMetrics.synthetic_accessibility_score_scores(mol, norm=True)

        # np.array([m, n, j])  # row 1=m, row 2=n row 3=j
        y_arr = np.array([druglikeness, solubility, synthesizability])

    y_arr = torch.from_numpy(y_arr)
    # y_arr.to(dtype=torch.float32)

    if reward_type == 'druglikeness' or reward_type == 'synthesizability' or reward_type == 'solubility' or reward_type == 'valid':
        return y_arr[0].to(torch.float32)
    elif reward_type == 'joint':
        return y_arr[:, 0].to(torch.float32)


def drop_non_sanitizables_get_mol(dataset):
    # from original data, we drop mols non sanitizable (to avoid problems when computing targets for reward net)
    # this returns a dataset (list of pyg objs) and the mols for every mol in dataset, at the same idx
    mols_ = []
    indices_to_drop = []

    for i, obs in enumerate(dataset):

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
        try:
            mols_.append(numpy_to_rdkit(adj, nf, ef, sanitize=True))
        except:
            indices_to_drop.append(i)

    idxs_to_be_dropped = set(indices_to_drop)
    len_ = len(dataset)

    dataset_ = []
    for idx in range(len_):
        if idx not in idxs_to_be_dropped:
            dataset_.append(dataset[idx])

    return dataset_, mols_


def get_mol(obs):
    obs = obs[0]
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
    try:
        mol = numpy_to_rdkit(adj, nf, ef, sanitize=True)
        return [mol]
    except:
        return []
