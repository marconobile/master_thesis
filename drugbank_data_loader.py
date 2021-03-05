# final INPUT DATA PIPELINE:

import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)
import numpy as np

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


#############################

def load_drugbak_data():
    # final INPUT DATA PIPELINE:

    import os
    import os.path as osp

    import torch
    import torch.nn.functional as F
    from torch_sparse import coalesce
    from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                      Data)
    import numpy as np
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

    #############################

    # FIRST LOAD ALL THE DATA, REMOVING Hs
    suppl = Chem.SDMolSupplier('./structures_ALL.sdf', removeHs=True)

    # STEP 1: drop all mols with n. of atoms == 1 and >100:
    suppl_filtered_1 = []
    for mol in suppl:
        if mol:
            if mol.GetNumAtoms() != 1 and mol.GetNumAtoms() <= 100:
                suppl_filtered_1.append(mol)

    # STEP 2: drop all mols with explicit Hs
    suppl_filtered_2 = []
    for mol in suppl_filtered_1:
        if "H" not in [atom.GetSymbol() for atom in mol.GetAtoms()]:
            suppl_filtered_2.append(mol)

    # STEP 2 bis: drop all mols with disconnected fragments:
    suppl_filtered_3 = []
    for mol in suppl_filtered_2:
        if len(Chem.rdmolops.GetMolFrags(mol)) == 1:
            suppl_filtered_3.append(mol)

    # STEP 3: after removing all mols with explicit Hs, get node type dict
    list_of_atoms = []
    for mol in suppl_filtered_3:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in list_of_atoms:
                list_of_atoms.append(atom.GetSymbol())

    types_dict = {}
    for i in range(len(list_of_atoms)):
        types_dict[list_of_atoms[i]] = i
    print(types_dict)
    # STEP 4: cast to PyG
    types = types_dict
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    data_list = []
    for i, mol in enumerate(suppl_filtered_3):
        if mol is None:
            continue

        N = mol.GetNumAtoms()
        #     print('ID mol: ',i,' Number of atoms: ', N, [atom.GetSymbol() for atom in mol.GetAtoms()])

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

    # done, nb: use clean_dataset(data_list) to drop mols with disconnected components
    # IMPORTANT, AFTER THE FILTERING WE HAVE ONLY 47 NODE TYPES


