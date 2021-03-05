#!/home/marcon/miniconda3/bin/python

import warnings
import logging
import os
import random
import torch
import numpy as np
from data_loader import QM9
from data_utils import clean_dataset, get_qm9_smiles
from loggers_setup import setup_logger
import matplotlib as mpl
from get_qm9_5k_subset import get_qm9_5k_subset
from args import Args
from random import shuffle
# from new_method import new_method
from new_method_with_RL import new_method
from drugbank_data_loader import load_drugbak_data
mpl.use('Agg')
warnings.filterwarnings("ignore")

from supervised_generation import supervised_generation

# 5k subset of mols of QM9 taken from:
# https://github.com/gablg1/ORGAN/blob/master/data/qm9_5k.csv

if __name__ == "__main__":

    args = Args()

    print("Setting up loggers, seed and device...")
    setup_logger('train_loss_log', r'./train_loss_log')
    setup_logger('test_loss_log', r'./test_loss_log')

    train_log = logging.getLogger('train_loss_log')
    test_log = logging.getLogger('test_loss_log')

    setup_logger('weights_log', r'./weights_log')
    weights_log = logging.getLogger('weights_log')

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

    print('Set up executed. Starting data loading...')


    if args.drugbank==True:
        dataset = load_drugbak_data()
    elif args.QM9_dataset == True:
        cwd = os.getcwd()
        if args.all_data:
            dataset = QM9(root=cwd + '/data', transform=None, pre_transform=None, pre_filter=None)

            print('Loaded ALL data')
        else:
            dataset = get_qm9_5k_subset('./qm9_5k_smiles.txt')

            print('Loaded 5k subset')

#    print("Len of dataset pre clean_dataset", len(dataset))
    dataset = clean_dataset(dataset)
#    print("Len of dataset post clean_dataset", len(dataset))


    qm9_smiles = get_qm9_smiles(dataset) # smiles of the dataset in input
    print("Dataset len", len(dataset))

    print('Dataset loaded, cleaned and got original data in chem.mols objs')

    with_absence_net = args.with_absence_net

    # dataset = dataset[:320]
    if args.new_method:
        shuffle(dataset)
        new_method(dataset, qm9_smiles, cuda, device, args)
 
    # if args.supervised:
    #     supervised_generation(dataset, with_absence_net, device, cuda, train_log, test_log, qm9_smiles)




