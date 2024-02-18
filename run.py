from functools import partial
import torch.nn as nn
from model import get_generator
from supervised_tools.create_train_val_data import create_train_val_dataloaders
import numpy as np
import os
from torch_geometric.data import Data
import torch.nn.functional as F
from rdkit import Chem
# from utils.setup import setup
import torch
print(torch.__version__)
import matplotlib.pyplot as plt
# from torch_lr_finder import LRFinder
from utils.data_utils import mols_from_file, get_atoms_info, rdkit2pyg, pyg2rdkit, save_smiles
# from mappings import *
from train_functions import *


#! --- GET DATA ---
guacm_smiles = "/home/nobilm@usi.ch/master_thesis/guacamol/testdata.smiles"
train_guac_mols = mols_from_file(guacm_smiles, True)
train_data = rdkit2pyg(train_guac_mols[:32])


#! --- GET WEIGHTS ---
nweights = {
    'C':    0.03238897867833534,
    'Br':   14.044943820224718,
    'N':    0.21620219229022983,
    'O':    0.2177273617975571,
    'S':    1.6680567139282736,
    'Cl':   2.872737719046251,
    'F':    1.754693805930865,
    'P':    37.735849056603776,
    'I':    100.0,
    'B':    416.6666666666667,
    'Si':   454.54545454545456,
    'Se':   833.3333333333334
}
bweights = { 
    BT.SINGLE:      4.663287337775892, 
    BT.AROMATIC:    4.77780803722868, 
    BT.DOUBLE:      34.74514436607484, 
    BT.TRIPLE:      969.9321047526673 
}

nweights_list = [nweights[k] for k in atom2num]
bweights_list = [bweights[k] for k in bond2num]
bweights_list.insert(0, 1500)
node_weights = torch.tensor(nweights_list) 
edge_weights = torch.tensor(bweights_list) 

#! --- SET UP EXPERIMENT ---
LRrnn, LRout = 1e-5, 1e-5
wd = 5e-4
epoch, max_epoch = 1, 5000
bs = 32

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")

train_dataset_loader, val_dataset_loader = create_train_val_dataloaders(train_data, train_data, max_num_node, max_prev_node, bs) #! HERE WORKERS
rnn, output = get_generator()
rnn.apply(weight_init)
output.apply(weight_init)
rnn.ad_hoc_init()
output.ad_hoc_init()

params = list(rnn.parameters()) + list(output.parameters())
optimizer = torch.optim.RMSprop(params, lr=LRrnn)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LRrnn, steps_per_epoch=len(train_dataset_loader), epochs=max_epoch)


# # # MEMORIZATION
# # obs = train_guac_mols[5]
# # print(Chem.MolToSmiles(obs))
# # train_data = rdkit2pyg([obs])
# train_dataset_loader, val_dataset_loader = create_train_val_dataloaders(train_data, train_data, max_num_node, max_prev_node, bs) #! HERE WORKERS
# max_epoch = 5000
# params = list(rnn.parameters()) + list(output.parameters())
# optimizer = torch.optim.RMSprop(params, lr=LRrnn)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LRrnn, steps_per_epoch=len(train_dataset_loader), epochs=max_epoch)
# # memorize_batch_single_opt(max_epoch, rnn, output, train_dataset_loader, optimizer, node_weights, edge_weights, scheduler)



# # # GRADIENTS
grads_dict = {}
for name, module in rnn.named_modules():
    for pname, p in module.named_parameters():
        if ".weight" in pname:
            grads_dict[pname] = []

@torch.no_grad()
def get_grads():
    for name, module in rnn.named_modules():
        for pname, p in module.named_parameters():
            if ".weight" in pname:
                grads_dict[pname].append(p.grad.data)



epoch = 0
update = []
while epoch < max_epoch:
    rnn.train()
    output.train()
    loss_all, loss_sum_edges, loss_sum_nodes = 0, 0, 0
    for batch_idx, data in enumerate(train_dataset_loader):
        rnn.zero_grad()
        output.zero_grad()
        loss, edge_loss, node_loss = fit_batch(data, rnn, output, node_weights, edge_weights)
        loss.backward()
        get_grads()
        if scheduler != None: 
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr'][0]
        
        with torch.no_grad():
            update.append([(current_lr* p.grad.std()/p.data.std()).log10().item() for p in rnn.parameters()])
        
        optimizer.step()        
        if scheduler != None: scheduler.step()
        loss_sum_edges += edge_loss.data
        loss_sum_nodes += node_loss.data
        loss_all =  loss_sum_edges + loss_sum_nodes
        print(f"Epoch {epoch}, ", loss / (batch_idx + 1), 'lossedges', loss_sum_edges / (batch_idx + 1), ' lossnodes ',loss_sum_nodes / (batch_idx + 1))
    epoch +=1

