from data_utils import get_qm9_smiles
from get_qm9_5k_subset import get_qm9_5k_subset
from create_train_val_data import create_train_val_dataloaders_geometric
from reward_structure import drop_non_sanitizable,  get_mol, set_reward
from disc_model import Reward_Net_Single #,Reward_Net_Joint
import torch
from loggers_setup import setup_logger
import logging
import numpy as np
import random
import os
from data_utils import get_qm9_smiles
from rdkit import Chem
from drugbank_data_loader import load_drugbak_data
from args import Args
from chem import validate_rdkit_mol
from new_method import pyg_to_rdkit


# SEEDS
manualSeed = 23742
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
if cuda:
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# LOAD DATA:
args = Args()
dataset = load_drugbak_data()
print('Loaded drugbank data,Dataset len: ', len(dataset))

# get smiles
drugbank_smiles = get_qm9_smiles(dataset)
rdkit_mols =[]

for i, mol in enumerate(dataset):
    rdkit_mols.append(pyg_to_rdkit(mol))

print('Dataset loaded, cleaned and got original data in chem.mols objs')
dataset = set_reward(dataset, rdkit_mols, reward_type='valid')  # dataset with labels
print("Dataset len after dropping non santizable mols from input data:", len(dataset))
train_dataset_loader, test_dataset_loader = create_train_val_dataloaders_geometric(dataset)

# 8750 valid mols, 1800 non valid mols

setup_logger('reward_net_on_original_data', r'./reward_net_on_original_data')
reward_net_on_original_data = logging.getLogger('reward_net_on_original_data')

reward_net = Reward_Net_Single(args)
if cuda:
    reward_net.to(device)
print(reward_net)
reward_net.train()
LR = 1e-4
optimizer_reward_net = torch.optim.Adam(list(reward_net.parameters()), lr=LR, weight_decay=5e-5)
max_num_of_epochs = 10000
for epoch in range(max_num_of_epochs):
    loss = 0
    for i, data in enumerate(train_dataset_loader, 0):
        data.to(device)
        reward_net.zero_grad()
        output_reward_original = reward_net(data).view(-1)
        loss_real_rewards = torch.mean((output_reward_original - data.y) ** 2)
        loss_real_rewards.backward()
        optimizer_reward_net.step()
        # print("Loss:", loss_real_rewards.item())
        loss += loss_real_rewards.item()
    reward_net_on_original_data.info(loss/len(train_dataset_loader))


##
path = os.getcwd() + "/weights/"  # "/content/gdrive/My Drive/GraphRNN_weights/"
checkpoint_reward_net = {
    'model_state_dict': reward_net.state_dict(),
    # 'epoch': epoch,
    # 'optimizer_rnn': optimizer_rnn.state_dict(),
    # 'scheduler_rnn': scheduler_rnn.state_dict()
}

torch.save(checkpoint_reward_net, path + f'reward_net_checkpoint_{epoch}.pth')

