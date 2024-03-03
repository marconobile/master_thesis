from rdkit.Chem.rdchem import BondType as BT

max_num_node = 10
max_prev_node = max_num_node - 1

node_feature_dims = 12
edge_feature_dims = 5

atom2num = {'Cl': 0, 'B': 1, 'F': 2, 'Si': 3, 'N': 4, 'S': 5, 'I': 6, 'O': 7, 'P': 8, 'Br': 9, 'Se': 10, 'C': 11}
bond2num = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
num2atom = {v:k for k,v in atom2num.items()}
num2bond = {v: k for k, v in bond2num.items()}

nweights = {
    'C':    1.,  #03805769363678407,  # 1.0,#
    'Br':   16.347081228646626,  # 1.0,#
    'N':    1.2521717662942678,  # 1.0,#
    'O':    1.2571374939894111,  # 1.0,#
    'S':    1.9962749509415432,  # 1.0,#
    'Cl':   3.4266994716029413,  # 1.0,#
    'F':    2.004610604390097,  # 1.0,#
    'P':    46.948356807511736,  # 1.0,#
    'I':    122.63919548687761,  # 1.0,#
    'B':    469.4835680751174,  # 1.0,#
    'Si':   519.2107995846313,  # 1.0,#
    'Se':   713.26676176890164  # 1.0,#
}


bweights = {
    BT.SINGLE: 1.5474413794296493,
    BT.AROMATIC: 1.561455769468956,
    BT.DOUBLE: 4.079857498737284,
    BT.TRIPLE: 117.53505483010308
}

nweights_list = [nweights[k] for k in atom2num]
bweights_list = [bweights[k] for k in bond2num]
bweights_list.insert(0, 200.) # 500