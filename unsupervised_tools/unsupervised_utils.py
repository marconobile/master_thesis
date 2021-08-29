import os
from random import shuffle
from itertools import islice
from torch_geometric.data import Data
import torch
from supervised_tools.generate_single_obs import generate_single_obs
from copy import deepcopy
from unsupervised_tools.LineGraph_transform import LineGraph
from utils.data_utils import mols_txt, mols_smiles_plots
from unsupervised_tools.save_gan_generator import save_model
from rdkit import Chem


def clamp_params(network):
    for p in network.parameters():
        p.data.clamp_(-0.01, 0.01)


def split_every(n, iterable):
    shuffle(iterable)
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


def get_fake_obs_via_ECC(ECC_nodes, ECC_edges, g_nodes, g_edges, epoch):
    edg_idx = g_nodes.edge_index

    new_node_matrix = ECC_nodes(g_nodes, epoch)
    new_edge_matrix = ECC_edges(g_edges, epoch)
    new_edge_matrix_1 = new_edge_matrix[g_edges.edge_map]

    return Data(x=new_node_matrix.to(torch.float32), edge_index=edg_idx.to(torch.long),
                edge_attr=new_edge_matrix_1.to(torch.float32))


def plot_graphs(rnn, output, absence_net, ECC_nodes, ECC_edges, epoch, device, args, smiles_list):
    print('starting generation')

    number_of_observations = 6400
    list_for_transf = []
    list_of_observations = []  # list of pyg obs
    while (len(list_of_observations) != number_of_observations):
        obs = generate_single_obs(rnn, output, absence_net, device, args=args, test_batch_size=1,
                                  max_num_node=args.max_num_node,
                                  max_prev_node=args.max_prev_node)

        obs = obs.coalesce()
        list_of_observations.append(obs)
        obs_1 = deepcopy(obs)
        list_for_transf.append(obs_1)

    line_graphs_for_list_of_observations = []
    transf = LineGraph()
    for g in list_for_transf:
        line_graphs_for_list_of_observations.append(transf(g))

    fake_datalist = []
    for g_nodes, g_edges in zip(list_of_observations, line_graphs_for_list_of_observations):
        with torch.no_grad():
            ECC_nodes.eval()
            ECC_edges.eval()
            data_graph = get_fake_obs_via_ECC(ECC_nodes, ECC_edges, g_nodes, g_edges, epoch)
        fake_datalist.append(data_graph)

    rdkit_mols, mols_smiles = mols_smiles_plots(fake_datalist, './figures/FIG_epoch_' + str(epoch))
    mols_txt(epoch, rdkit_mols, mols_smiles, smiles_list)
    print('data generated plotted')


def generation_step_and_save(epoch, ECC_nodes, ECC_edges, critic, rnn, output, absence_net, device, args,
                             smiles_list):
    if epoch >= 100 and epoch % 100 == 0:
        save_model(ECC_nodes, ECC_edges, epoch)

        path = os.getcwd() + "/weights/"
        checkpoint_critic = {'model_state_dict': critic.state_dict()}
        torch.save(checkpoint_critic, path + f'critic_checkpoint_{epoch}.pth')

        plot_graphs(rnn, output, absence_net, ECC_nodes, ECC_edges, epoch, device, args, smiles_list)
    return None


def filter_obs_by_ring_size(mol):
    '''Given a rdkit mol, filter_obs(mol) returns true if the mol has rings of size 5 or 6
    '''

    try:
        # Chem.SanitizeMol(mol)
        ring_info = mol.GetRingInfo()
        n_of_rings = ring_info.NumRings()
    except:
        return False

    if n_of_rings == 0:
        return False

    list_size_of_rings = []
    list_of_atom_rings = ring_info.AtomRings()
    for i in list_of_atom_rings:
        list_size_of_rings.append(len(i))

    # filter for ring size:
    switch = True
    for ring_dim in list_size_of_rings:
        if (ring_dim != 5) and (ring_dim != 6):
            switch = False

    return switch
