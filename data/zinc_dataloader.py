import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import Data

try:
    import rdkit
    from rdkit import Chem
    from rdkit import rdBase
    from rdkit.Chem.rdchem import HybridizationType
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem.rdchem import BondType as BT

    rdBase.DisableLog('rdApp.error')
except ImportError:
    rdkit = None


def load_zinc(zinc_mols):
    '''
    Takes as input a list of molecules (either ../data/ZINC_mols_train or ZINC_mols_test) and
    returns a list of PyG observations already one-hot encoded
    '''

    types = {'C': 0,
             'O': 1,
             'N': 2,
             'F': 3,
             'S': 4,
             'Cl': 5,
             'Br': 6,
             'I': 7,
             'P': 8}

    bonds = {BT.SINGLE: 0,
             BT.DOUBLE: 1,
             BT.TRIPLE: 2}

    data_list = []

    for i, mol in enumerate(zinc_mols):
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
