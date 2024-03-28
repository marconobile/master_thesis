#!/home/marcon/miniconda3/bin/python
import warnings

warnings.filterwarnings("ignore")

import os
import pickle
from utils.setup import setup
from args import Args
from utils.data_utils import get_smiles
from data.zinc_dataloader import load_zinc
from random import shuffle
from supervised_tools.supervised_training import supervised_training

from rdkit import Chem


from utils.data_utils import mols_from_file, rdkit2pyg


if __name__ == "__main__":

    args = Args()
    device, cuda, train_log, test_log = setup()
    cwd = os.getcwd()

    # if args.ZINC_dataset == True:
    #     if args.ZINC_filtered == True:
    #         with open(cwd + '/data/NEW_ZINC_FILTERED', 'rb') as fp:
    #             zinc_mols = pickle.load(fp)

    #     dataset = load_zinc(zinc_mols)  # pyg dataloader
    #     dataset_smiles = get_smiles(dataset)
    #     shuffle(dataset)

    #     dataset = dataset[:1000]

    #     if args.test_set == True:
    #         with open(cwd + '/data/ZINC_mols_test', 'rb') as fp:
    #             zinc_mols_test = pickle.load(fp)
    #             dataset_test = load_zinc(zinc_mols_test)
    #     else:
    #         dataset_test = False

    path = "/home/nobilm@usi.ch/master_thesis/guacamol/testdata.smiles"
    mols = mols_from_file(path, True)
    dataset = rdkit2pyg(mols)
    OBS_IDX = 5
    dataset = [dataset[OBS_IDX]]


    # if args.supervised:
    supervised_training(dataset, device, cuda, train_log, test_log, None, None)
    print(Chem.MolToSmiles(mols[OBS_IDX]))