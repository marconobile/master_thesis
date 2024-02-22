from rdkit.Chem.rdchem import BondType as BT

max_num_node = 88
max_prev_node = max_num_node - 1


node_feature_dims = 12
edge_feature_dims = 5


atom2num = {'Cl': 0, 'B': 1, 'F': 2, 'Si': 3, 'N': 4, 'S': 5, 'I': 6, 'O': 7, 'P': 8, 'Br': 9, 'Se': 10, 'C': 11}
bond2num = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
num2atom = {v:k for k,v in atom2num.items()}
num2bond = {v: k for k, v in bond2num.items()}

nweights = {
    'C':    1.0,#1000.03805769363678407,
    'Br':   1.0,#16.347081228646626,
    'N':    1.0,#1.2521717662942678,
    'O':    1.0,#1.2571374939894111,
    'S':    1.0,#1.9962749509415432,
    'Cl':   1.0,#3.4266994716029413,
    'F':    1.0,#2.004610604390097,
    'P':    1.0,#46.948356807511736,
    'I':    1.0,#122.63919548687761,
    'B':    1.0,#469.4835680751174,
    'Si':   1.0,#519.2107995846313,
    'Se':   1.0,#713.26676176890164
}


bweights = {
    BT.SINGLE: 1.0,# 1.5474413794296493,
    BT.AROMATIC: 1.0,# 1.561455769468956,
    BT.DOUBLE: 1.0,# 4.079857498737284,
    BT.TRIPLE: 1.0,# 117.53505483010308
}

nweights_list = [nweights[k] for k in atom2num]
bweights_list = [bweights[k] for k in bond2num]
bweights_list.insert(0, 500.) # 500