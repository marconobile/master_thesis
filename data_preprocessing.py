import os
import pickle
from data.zinc_dataloader import load_zinc
from utils.data_utils import get_smiles
from rdkit import Chem

cwd = os.getcwd()
with open(cwd + '/data/ZINC_mols_train', 'rb') as fp:
    zinc_mols = pickle.load(fp)

dataset = load_zinc(zinc_mols)
dataset_smiles = get_smiles(dataset)

def isRingAromatic(mol, bondRing):
    for id in bondRing:
        if not mol.GetBondWithIdx(id).GetIsAromatic():
            return False
    return True    
    

# Filters:
# 1) molecular weight
# 2) n of atoms in rings == 5/6
# 3) aromatic bonds mandatory
# 4) rings have to be aromatic

# cast to mols from dataset_smiles
mols = []
for smi in dataset_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        mols.append(mol)

# 1) molecular weight
# filter out all mols with mol weight more than 500 dalton
suppl_filtered_1 = []
for mol in mols:
    if Chem.rdMolDescriptors.CalcExactMolWt(mol)<=500:
        suppl_filtered_1.append(mol)
        
# 2) n of atoms in rings == 5/6
suppl_filtered_2 = [] 
for mol in suppl_filtered_1:    
    ring_sizes = []
    n_of_rings_ = mol.GetRingInfo()    

    for i in n_of_rings_.AtomRings():
        ring_sizes.append(len(i))
    
    check = True
    for r_s in ring_sizes:
        if (r_s != 5 and r_s !=6):
            check = False
    if check:
        suppl_filtered_2.append(mol)
        
# 3) aromatic bonds mandatory
suppl_filtered_3 = []
for mol in suppl_filtered_2:
    check = False
    for bond in mol.GetBonds():
        if (str(bond.GetBondType()) == 'AROMATIC'):
            check = True
    if check:
        suppl_filtered_3.append(mol)
        
# 4) rings have to be aromatic
# to select only mols with only arom rings
suppl_filtered_4 = []
for mol in suppl_filtered_3:
    n_of_rings_ = mol.GetRingInfo() 
    
    ri_list = []

    for j,ring in enumerate(n_of_rings_.BondRings()):
        ri_list.append(isRingAromatic(mol, n_of_rings_.BondRings()[j]))
    if all(ri_list):
        suppl_filtered_4.append(mol)
    
# saving to file
# with open('NEW_ZINC_FILTERED', 'wb') as fp:
#     pickle.dump(suppl_filtered_4, fp)

# load from file
# print('Loading rdkit.mols from file:')
# with open('/NEW_ZINC_FILTERED', 'rb') as fp:
#     list_of_observations = pickle.load(fp)


    



