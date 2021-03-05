from new_gen import new_gen
from new_method import pyg_to_rdkit,LineGraph
import numpy as np
from rdkit import Chem
# from copy import deepcopy
from data_utils import mols_smiles_plots, mols_txt

def plot_graphs_no_chains(rnn, output, absence_net, epoch, device, args, qm9_smiles):

    print('starting generation')

    number_of_observations = 6400
    list_of_observations = []  # list of pyg obs
    while (len(list_of_observations) != number_of_observations):
        obs = new_gen(rnn, output, absence_net, device, args=args, test_batch_size=1, max_num_node=args.max_num_node,
                      max_prev_node=args.max_prev_node)
        m = pyg_to_rdkit(obs)
        if (obs.x.shape[0] > 2) and (m != None):

            adj_m_ = np.array(Chem.GetAdjacencyMatrix(m), "d")
            if max(sum(adj_m_)) != 1 and max(sum(adj_m_)) != 2:  # filter out chains

                list_of_observations.append(obs)

    print("Second run of generation done for epoch", epoch)
    rdkit_mols, mols_smiles = mols_smiles_plots(list_of_observations, './figures_no_chains/FIG_epoch_' + str(epoch), epoch)
    mols_txt(epoch, rdkit_mols, mols_smiles, qm9_smiles)
    print('data generated plotted version B')
