import os
import pickle
import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.utils import to_dense_adj
from rdkit import Chem as rdc

# BOND_MAP and NUM_TO_SYMBOL encodings defined as in /PYG_ZINC/README.txt
BOND_MAP = {  # skip 0 since it represents absence of connection between node i,j in adjacency matrix
    1: rdc.rdchem.BondType.SINGLE,
    2: rdc.rdchem.BondType.DOUBLE,
    3: rdc.rdchem.BondType.TRIPLE,
}

NUM_TO_SYMBOL = {0: 'C',
                 1: 'O',
                 2: 'N',
                 3: 'F',
                 4: 'C',
                 5: 'S',
                 6: 'Cl',
                 7: 'O',
                 8: 'N',
                 9: 'Br',
                 10: 'N',
                 11: 'N',
                 12: 'N',
                 13: 'N',
                 14: 'S',
                 15: 'I',
                 16: 'P',
                 17: 'O',
                 18: 'N',
                 19: 'O',
                 20: 'S',
                 21: 'P',
                 22: 'P',
                 23: 'C',
                 24: 'P',
                 25: 'S',
                 26: 'C',
                 27: 'P'}


def numpy_to_rdkit_(adj, nf, sanitize=False):
    """
    Function modified from chem.py
    Converts a molecule from numpy to RDKit format.
    :param adj: binary numpy array of shape (N, N) whose entry i,j is the non-encoded edge class
    :param nf: numpy array of shape (N, F)
    :param sanitize: whether to sanitize the molecule after conversion
    :return: an RDKit molecule
    """
    if rdc is None:
        raise ImportError('`numpy_to_rdkit` requires RDKit.')
    mol = rdc.RWMol()
    for nf_ in nf:
        atomic_num = int(nf_)
        mol.AddAtom(rdc.Atom(NUM_TO_SYMBOL[atomic_num]))

    for i, j in zip(*np.triu_indices(adj.shape[-1])):
        if i != j and adj[i, j] == adj[j, i] != 0 and not mol.GetBondBetweenAtoms(int(i), int(j)):
            bond_type_1 = BOND_MAP[int(adj[i, j].item())]
            bond_type_2 = BOND_MAP[int(adj[j, i].item())]
            if bond_type_1 == bond_type_2:
                mol.AddBond(int(i), int(j), bond_type_1)

    mol = mol.GetMol()
    if sanitize:
        rdc.SanitizeMol(mol)
    return mol


def main():
    '''
    Open ZINC train dataset and retrieves its RDKit mols
    with encoding defined as in /PYG_ZINC/README.txt
    '''

    cwd = os.getcwd()
    # repeat for train.pickle and for test.pickle
    with open(cwd + '/PYG_ZINC/molecules/test.pickle', 'rb') as fp:
        mols = pickle.load(fp)  # molecules defined as list
        # eg:
        # {'num_atom': 33,
        #  'atom_type': tensor([0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
        #                       0, 0, 0, 0, 0, 1, 2, 0, 5], dtype=torch.int8),
        #  'bond_type': tensor([[0, 1, 0, ..., 0, 0, 0],
        #                       [1, 0, 1, ..., 0, 0, 0],
        #                       [0, 1, 0, ..., 0, 0, 0],
        #                       ...,
        #                       [0, 0, 0, ..., 0, 2, 0],
        #                       [0, 0, 0, ..., 2, 0, 1],
        #                       [0, 0, 0, ..., 0, 1, 0]], dtype=torch.int8),
        #  'logP_SA_cycle_normalized': tensor([3.0464])}

    indices = range(len(mols))

    data_list = []
    for idx in indices:
        mol = mols[idx]

        x = mol['atom_type'].to(torch.long).view(-1, 1)

        adj = mol['bond_type']
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        data_list.append(data)

    # from PyG data loader encoding, to mol object such to apply the encoding required for the
    # experiments here proposed (encoding from mols to PyG Data obj applied in zinc_dataloader.py)
    mols_data_list = []
    for obs in data_list:
        ef_temp = torch.squeeze(to_dense_adj(edge_index=obs.edge_index, batch=None, edge_attr=obs.edge_attr), 0)
        mols_data_list.append(numpy_to_rdkit_(ef_temp, obs.x, sanitize=False))

	# ZINC_mols_train or ZINC_mols_test accordingly
    with open('ZINC_mols_test', 'wb') as fp: 
        pickle.dump(mols_data_list, fp)


main()
