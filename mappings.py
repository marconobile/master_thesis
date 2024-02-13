from rdkit.Chem.rdchem import BondType as BT

max_num_node = 88
node_feature_dims = 12
edge_feature_dims = 5
max_prev_node = max_num_node - 1

atom2num = {'Cl': 0, 'B': 1, 'F': 2, 'Si': 3, 'N': 4, 'S': 5, 'I': 6, 'O': 7, 'P': 8, 'Br': 9, 'Se': 10, 'C': 11}
bond2num = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
num2atom = {v:k for k,v in atom2num.items()}
num2bond = {v: k for k, v in bond2num.items()}