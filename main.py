#!/home/marcon/miniconda3/bin/python

import warnings

warnings.filterwarnings("ignore")

import os
import pickle
from utils.setup import setup
from args import Args
from utils.data_utils import get_smiles
from data.zinc_dataloader import load_zinc

from supervised_tools.supervised_training import supervised_training
from unsupervised_tools.unsupervised_training import unsupervised_training

if __name__ == "__main__":

    args = Args()
    device, cuda, train_log, test_log = setup()
    cwd = os.getcwd()

    if args.ZINC_dataset == True:
        with open(cwd + '/data/ZINC_mols_train', 'rb') as fp:
            zinc_mols = pickle.load(fp)
        dataset = load_zinc(zinc_mols)
        dataset_smiles = get_smiles(dataset)

        if args.test_set == True:
            with open(cwd + '/data/ZINC_mols_test', 'rb') as fp:
                zinc_mols_test = pickle.load(fp)
                dataset_test = load_zinc(zinc_mols_test)
        else:
            dataset_test = False

    if args.supervised:
        supervised_training(dataset, device, cuda, train_log, test_log, dataset_smiles, dataset_test)
    else:
        unsupervised_training(dataset, dataset_smiles, cuda, device, args)  # , train_with_RL)
