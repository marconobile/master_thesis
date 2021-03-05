from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
import torch.nn.functional as F
import torch
from torch_sparse import coalesce
from torch_geometric.data import Data


def get_qm9_5k_subset(path_to_file):
    # file contains SMILES, so cast to rdkit.mols
    list_of_mols = []
    with open(path_to_file, 'r') as f:
        for line in f:
            list_of_mols.append(Chem.MolFromSmiles(str(line.strip('\n')))) # FALSE

    types = {'C': 0, 'N': 1, 'O': 2, 'F': 3}  # 'H': 0,
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    data_list = []
    for i, mol in enumerate(list_of_mols):
        if mol is None:
            continue

        N = mol.GetNumAtoms()

        type_idx = []

        for atom in mol.GetAtoms():
            type_idx.append(types[atom.GetSymbol()])

        x = F.one_hot(torch.tensor(type_idx), num_classes=len(types))

        row, col, bond_idx = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            bond_idx += 2 * [bonds[bond.GetBondType()]]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = F.one_hot(torch.tensor(bond_idx).to(torch.int64),
                              num_classes=len(bonds)).to(torch.float)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        data_list.append(data)

    return data_list
