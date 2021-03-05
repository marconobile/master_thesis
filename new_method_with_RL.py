from get_generator import get_generator
from new_gen import new_gen
import torch
from data_utils import mols_smiles_plots, mols_txt
from torch_geometric.utils import to_dense_adj
import numpy as np
from chem import numpy_to_rdkit
import torch.nn.functional as F
import torch.nn as nn
from disc_model import Critic
from torch_geometric.data import Data
import torch_geometric
from torch_geometric.nn.conv import NNConv
from itertools import islice
from torch_sparse import coalesce
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import remove_self_loops
from copy import deepcopy
import os
import logging
from loggers_setup import setup_logger
from rdkit import Chem
from reward_structure import set_reward
from disc_model import Reward_Net_Single
import pickle


def save_model(ECC_nodes, ECC_edges, epoch):
    path = os.getcwd() + "/weights/"
    checkpoint_ecc_nodes = {
        'model_state_dict': ECC_nodes.state_dict(),
    }
    torch.save(checkpoint_ecc_nodes, path + f'ECC_nodes_checkpoint_{epoch}.pth')

    checkpoint_ecc_edges = {
        'model_state_dict': ECC_edges.state_dict(),
    }

    torch.save(checkpoint_ecc_edges, path + f'ECC_edges_checkpoint_{epoch}.pth')
    print(f'Model saved at epoch {epoch}')


class LineGraph(object):

    def __init__(self, force_directed=False):
        self.force_directed = force_directed

    def __call__(self, data):
        N = data.num_nodes  # gen num of nodes of the given graph
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr  # creates local 2 vars for the graph attrs
        (row, col), edge_attr = coalesce(edge_index, edge_attr, N, N, op='max')  # rimuove doppi
        # represents the edg atrs in COO format as:
        # sparse.coo_matrix((data, (row, col)), shape=(4, 4))

        # Compute node indices.
        mask = row < col  # upper triang
        row, col = row[mask], col[mask]  # tiene solo i true cme indx
        i = torch.arange(row.size(0), dtype=torch.long, device=row.device)  #

        (row, col), i = coalesce(
            torch.stack([
                torch.cat([row, col], dim=0),  # for simmetry
                torch.cat([col, row], dim=0)
            ], dim=0), torch.cat([i, i], dim=0), N, N)

        # Compute new edge indices according to `i`.
        count = scatter_add(torch.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes)
        # conta quante volte un nodo appare cme indice i # num di outgoing edges from a node
        joints = torch.split(i, count.tolist())

        def generate_grid(j, x):
            row = j.view(-1, 1).repeat(1, j.numel()).view(-1)
            col = j.repeat(j.numel())
            edge_attr = x.view(1, -1).repeat(col.numel(), 1)
            return torch.stack([row, col], dim=0), edge_attr

        joints, joints_edge_attr = list(zip(*[generate_grid(joint, xx) for joint, xx in zip(joints, x)]))
        joints = torch.cat(joints, dim=1)
        joints_edge_attr = torch.cat(joints_edge_attr, dim=0)
        joints, joints_edge_attr = remove_self_loops(joints, joints_edge_attr)
        N = row.size(0) // 2
        joints, joints_edge_attr = coalesce(joints, joints_edge_attr, N, N, op='max')

        if edge_attr is not None:
            data.x, _ = scatter_max(edge_attr, i, dim=0, dim_size=N)

        data.edge_index = joints
        data.num_nodes = edge_index.size(1) // 2

        data.edge_attr = joints_edge_attr
        data.edge_map = i
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


class Graph_ECC(nn.Module):
    def __init__(self, node_dims, edge_dims):
        super(Graph_ECC, self).__init__()
        nn1 = nn.Sequential(nn.Linear(edge_dims, 32), nn.LeakyReLU(), nn.Linear(32, node_dims * 64))
        self.conv1 = NNConv(node_dims, 64, nn1, aggr='mean', root_weight=True)

        nn2 = nn.Sequential(nn.Linear(edge_dims, 48), nn.LeakyReLU(), nn.Linear(48, (64 + node_dims) * 64))
        self.conv2 = NNConv((64 + node_dims), 64, nn2, aggr='mean', root_weight=True)

        nn3 = nn.Sequential(nn.Linear(edge_dims, 64), nn.LeakyReLU(), nn.Linear(64, (64 + node_dims) * node_dims))
        self.conv3 = NNConv((64 + node_dims), node_dims, nn3, aggr='mean', root_weight=True)

    def forward(self, data):
        # data1 = F.leaky_relu(self.conv1(data.x.to(torch.float32), data.edge_index, data.edge_attr))
        #
        # data1 = torch.cat([data1, data.x.to(torch.float32)], dim=-1)
        #
        # data2 = F.leaky_relu(self.conv2(data1, data.edge_index, data.edge_attr))
        #
        # data2 = torch.cat([data2, data.x.to(torch.float32)], dim=-1)
        #
        # data3 = F.leaky_relu(self.conv3(data2, data.edge_index, data.edge_attr))

        data1 = F.leaky_relu(self.conv1(data.x.to(torch.float32), data.edge_index, data.edge_attr))

        data1 = torch.cat([data1, data.x.to(torch.float32)], dim=-1)

        data2 = F.leaky_relu(self.conv2(data1, data.edge_index, data.edge_attr))

        data2 = torch.cat([data2, data.x.to(torch.float32)], dim=-1)

        data3 = F.leaky_relu(self.conv3(data2, data.edge_index, data.edge_attr))

        data4 = F.gumbel_softmax(data3, -1, hard=True)
        return data4


def load_nets(cuda, device, to_be_loaded):  # to_be_loaded = int(last_epoch)
    path = "./weights"

    rnn, output, absence_net = get_generator()

    PATH1 = path + f'/nodeRNN_checkpoint_{to_be_loaded}.pth'
    if cuda:
        checkpoint1 = torch.load(PATH1)
    else:
        checkpoint1 = torch.load(PATH1, map_location='cpu')
    rnn.load_state_dict(checkpoint1['model_state_dict'])

    PATH2 = path + f'/edgeRNN_checkpoint_{to_be_loaded}.pth'
    if cuda:
        checkpoint2 = torch.load(PATH2)
    else:
        checkpoint2 = torch.load(PATH2, map_location='cpu')
    output.load_state_dict(checkpoint2['model_state_dict'])

    PATH3 = path + f'/absenceRNN_checkpoint_{to_be_loaded}.pth'
    if cuda:
        checkpoint3 = torch.load(PATH3)
    else:
        checkpoint3 = torch.load(PATH3, map_location='cpu')
    absence_net.load_state_dict(checkpoint3['model_state_dict'])

    rnn.to(device)
    output.to(device)
    absence_net.to(device)

    rnn.eval()
    output.eval()
    absence_net.eval()

    return rnn, output, absence_net


def pyg_to_rdkit(obs):
    try:
        ef_temp = torch.squeeze(to_dense_adj(edge_index=obs.edge_index, batch=None, edge_attr=obs.edge_attr), 0)

        ef = torch.zeros((ef_temp.shape[0], ef_temp.shape[1], 1))
        adj = torch.zeros((ef_temp.shape[0], ef_temp.shape[1]))

        for row in range(ef_temp.shape[0]):
            for col in range(ef_temp.shape[1]):
                if int(torch.sum(ef_temp[row, col]).item()) != 0:
                    ef[row, col, 0] = torch.argmax(ef_temp[row, col]).item()
                    adj[row, col] = 1

        ef = np.array(ef)
        adj = np.array(adj)

        adj = adj.astype(int)
        ef = ef.astype(int)

        nf = np.array([torch.argmax(row).item() for row in obs.x])
        nf = np.expand_dims(nf, 1)
        nf = nf.astype(int)

        m = numpy_to_rdkit(adj, nf, ef, sanitize=False)
        return m

    except:
        return None


def get_fake_obs_via_ECC(ECC_nodes, ECC_edges, g_nodes, g_edges):
    edg_idx = g_nodes.edge_index

    new_node_matrix = ECC_nodes(g_nodes)
    new_edge_matrix = ECC_edges(g_edges)
    new_edge_matrix_1 = new_edge_matrix[g_edges.edge_map]

    return Data(x=new_node_matrix.to(torch.float32), edge_index=edg_idx.to(torch.long),
                edge_attr=new_edge_matrix_1.to(torch.float32))


def plot_graphs(rnn, output, absence_net, ECC_nodes, GCN_edges, epoch, device, args, qm9_smiles):
    print('starting generation')

    number_of_observations = 1000
    list_for_transf = []
    list_of_observations = []  # list of pyg obs
    while (len(list_of_observations) != number_of_observations):
        obs = new_gen(rnn, output, absence_net, device, args=args, test_batch_size=1, max_num_node=args.max_num_node,
                      max_prev_node=args.max_prev_node)
        m = pyg_to_rdkit(obs)

        # if (m != None): # now I filter also here

        if (obs.x.shape[0] > 2) and (m != None):

            adj_m_ = np.array(Chem.GetAdjacencyMatrix(m), "d")
            if max(sum(adj_m_)) != 1 and max(sum(adj_m_)) != 2:  # filter out chains
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
        fake_datalist.append(get_fake_obs_via_ECC(ECC_nodes, GCN_edges, g_nodes, g_edges))

    rdkit_mols, mols_smiles = mols_smiles_plots(fake_datalist, './figures/FIG_epoch_' + str(epoch), epoch)
    mols_txt(epoch, rdkit_mols, mols_smiles, qm9_smiles)
    print('data generated plotted')


def clamp_params(network):
    for p in network.parameters():
        p.data.clamp_(-0.01, 0.01)


def new_method(dataset, qm9_smiles, cuda, device, args):
    # so here I have dataset ()
    # the smiles of the dataset
    # args

    # beginning of RL code
    rdkit_mols = []
    for i, mol in enumerate(dataset):
        rdkit_mols.append(pyg_to_rdkit(mol))
    dataset = set_reward(dataset, rdkit_mols, reward_type='valid')  # dataset with labels
    # labels applied for the original dataset

    setup_logger('RL_loss_log', r'./RL_loss_log')
    RL_loss_log = logging.getLogger('RL_loss_log')

    setup_logger('accuracy_RL_loss_log', r'./accuracy_RL_loss_log')
    accuracy_RL_loss_log = logging.getLogger('accuracy_RL_loss_log')

    setup_logger('critic_loss_log', r'./critic_loss_log')
    critic_loss_log = logging.getLogger('critic_loss_log')
    setup_logger('generator_loss_log', r'./generator_loss_log')
    generator_loss_log = logging.getLogger('generator_loss_log')
    #    dataset = dataset[:50016 + 1]  # 32*1563=50016 batches

    #####################################
    # RNN MODELS AND DATASET GENERATION #
    #####################################
    # STEP 1: load RNNs models architectures and load weights
    # STEP 2: get X observations via RNNs, filtering for (VALID AND |N|>2)
    # STEP 3: plot 'new' data and gather .txt
    # STEP 4: define new 'modificator', MLP in this case
    # STEP 5: define discriminator (not changed)
    # STEP 6: define optimizers and lr
    # STEP 7: PROCESSING data for GAN:
    #   - gather datasets -> dataloaders both for REAL and FAKE data

    # STEP 1: load RNNs models architectures and load weights
    # loaded and setted to .eval()
    epoch_to_load = 300
    rnn, output, absence_net = load_nets(cuda, device, epoch_to_load)
    print(f'GraphRNN at epoch {epoch_to_load} loaded')
    # STEP 2: get X observations via RNNs
    # print(f'starting generation: generating {len(dataset)} mols')
    # number_of_observations = len(dataset)
    # list_for_transf = []
    # list_of_observations = []  # list of pyg obs
    # while (len(list_of_observations) != number_of_observations):
    #     obs = new_gen(rnn, output, absence_net, device, args=args, test_batch_size=1, max_num_node=args.max_num_node,
    #                   max_prev_node=args.max_prev_node)
    #     # filtering for mols who are not none
    #
    #     m = pyg_to_rdkit(obs)
    #
    #     if (obs.x.shape[0] > 2) and (m != None):
    #
    #         adj_m_ = np.array(Chem.GetAdjacencyMatrix(m), "d")
    #         if max(sum(adj_m_)) != 1 and max(sum(adj_m_)) != 2:  # filter out chains
    #
    #             obs = obs.coalesce()
    #             list_of_observations.append(obs)
    #             obs_1 = deepcopy(obs)
    #             list_for_transf.append(obs_1)
    # print('generation ended, plotting...')




    # saving to file
    # import pickle
    # with open('outfile', 'wb') as fp:
    #     pickle.dump(list_of_observations, fp)

    # load from file
    print('Loading mols from file:')
    with open('outfile', 'rb') as fp:
        list_of_observations = pickle.load(fp)

    list_for_transf = []

    for obs in list_of_observations:
        obs = obs.coalesce()
        obs_1 = deepcopy(obs)
        list_for_transf.append(obs_1)

    # STEP 3: PLOT 'EM & .txt
    rdkit_mols, mols_smiles = mols_smiles_plots(list_of_observations, './figures/dataset', -1)
    mols_txt(-1, rdkit_mols, mols_smiles, qm9_smiles)
    print('data plotted')

    # STEP 4: define new 'modificator'
    ECC_nodes = Graph_ECC(node_dims=39, edge_dims=4)
    ECC_edges = Graph_ECC(node_dims=4, edge_dims=39)

    # STEP 5: define discriminator (not changed)
    critic = Critic(args=args)
    reward_net = Reward_Net_Single(args)

    # STEP 6: define optimizers and lr
    LR = 1e-5
    optimizer_nodes = torch.optim.RMSprop(list(ECC_nodes.parameters()), lr=LR)
    optimizer_edges = torch.optim.RMSprop(list(ECC_edges.parameters()), lr=LR)
    optimizer_critic = torch.optim.RMSprop(list(critic.parameters()), lr=LR)
    optimizer_reward_net = torch.optim.Adam(list(reward_net.parameters()), lr=LR, weight_decay=5e-5)

    # STEP 7a: load original data:
    # train_dataset_loader, _ = create_train_val_dataloaders_geometric(dataset)

    # STEP 7b: load fake data:
    # - 2 sets: 1) list_of_observations 2) line graphs

    line_graphs_for_list_of_observations = []
    transf = LineGraph()
    for g in list_for_transf:
        line_graphs_for_list_of_observations.append(transf(g))

    print('Starting gan training')
    if cuda:
        critic.to(device)
        ECC_nodes.to(device)
        ECC_edges.to(device)
        reward_net.to(device)

    lambda_ = 0.5
    max_num_of_epochs = 5000
    rl_epoch = 100
    N_CRITIC_STEPS = 5
    for epoch in range(max_num_of_epochs):
        # have to reset at each epoch, otherwise they run out and training stops

        iter_fake_graphs = split_every(32, list_of_observations)
        iter_fake_line_graphs = split_every(32, line_graphs_for_list_of_observations)
        iter_real_data = split_every(32, dataset)

        crit_steps = 0

        for i in range(len(list_of_observations) // 32):

            real_batch_1 = next(iter_real_data)
            real_batch_2 = torch_geometric.data.DataLoader(real_batch_1, batch_size=32)
            real_batch_3 = iter(real_batch_2)
            real_batch_4 = real_batch_3.next()
            real_batch_4.to(device)

            nodes_batch_1 = next(iter_fake_graphs)

            edges_batch_1 = next(iter_fake_line_graphs)

            if crit_steps < N_CRITIC_STEPS:
                critic.train()
                critic.zero_grad()
                reward_net.train()
                reward_net.zero_grad()

                ECC_nodes.eval()
                ECC_edges.eval()

                clamp_params(critic)
                err_real = torch.mean(critic(real_batch_4))  # E[D(x)]

                output_reward_original = reward_net(real_batch_4).view(-1)
                loss_real_rewards = torch.mean((output_reward_original - real_batch_4.y) ** 2)

                fake_datalist = []
                for g_nodes, g_edges in zip(nodes_batch_1, edges_batch_1):
                    fake_datalist.append(get_fake_obs_via_ECC(ECC_nodes, ECC_edges, g_nodes, g_edges))

                rdkit_mols = []
                for i, mol in enumerate(fake_datalist):
                    rdkit_mols.append(pyg_to_rdkit(mol))

                fake_datalist = set_reward(fake_datalist, rdkit_mols, reward_type='valid')

                fake_dataloader_pyg = torch_geometric.data.DataLoader(fake_datalist, batch_size=32)
                fake_data_iterator = iter(fake_dataloader_pyg)
                batch_generated = fake_data_iterator.next()
                batch_generated.to(device)

                err_fake = torch.mean(critic(batch_generated))  # E[D(G(z))]

                output_reward_fake = reward_net(batch_generated).view(-1)
                loss_fake_rewards = torch.mean((output_reward_fake - batch_generated.y) ** 2)

                loss_reward_net = (loss_real_rewards + loss_fake_rewards)
                critic_loss = err_fake - err_real  # want this min
                critic_loss.backward(retain_graph=True)
                loss_reward_net.backward()
                optimizer_reward_net.step()
                optimizer_critic.step()
                accuracy_RL_loss_log.info(str(epoch) + ' , ' + str(loss_reward_net.item()))

                crit_steps += 1
                print('crit_steps', crit_steps)

            else:
                # ADD IF TO DELAY BEGINNING OF RL
                ECC_nodes.train()
                ECC_edges.train()
                ECC_nodes.zero_grad()
                ECC_edges.zero_grad()
                critic.eval()
                reward_net.eval()

                fake_datalist = []
                for g_nodes, g_edges in zip(nodes_batch_1, edges_batch_1):
                    fake_datalist.append(get_fake_obs_via_ECC(ECC_nodes, ECC_edges, g_nodes, g_edges))

                fake_dataloader_pyg = torch_geometric.data.DataLoader(fake_datalist, batch_size=32)
                fake_data_iterator = iter(fake_dataloader_pyg)
                batch_generated = fake_data_iterator.next()
                batch_generated.to(device)

                output_critic_fake = critic(batch_generated)
                generator_loss = -torch.mean(output_critic_fake)

                if epoch > rl_epoch:
                    output_reward_fake = reward_net(batch_generated)
                    rl_loss = -torch.mean(output_reward_fake)  # has to decreas in plot
                    loss = (lambda_ * generator_loss) + (1 - lambda_) * rl_loss
                    loss.backward()
                    RL_loss_log.info(str(epoch) + ' , ' + str(rl_loss.item()))
                else:
                    generator_loss = -torch.mean(output_critic_fake)
                    generator_loss.backward()

                optimizer_nodes.step()
                optimizer_edges.step()
                crit_steps = 0

                critic_loss_log.info(str(epoch) + ' , ' + str(critic_loss.item()))
                generator_loss_log.info(str(epoch) + ' , ' + str(generator_loss.item()))

                if epoch > rl_epoch:
                    print(
                    f"Epoch: {epoch}/{max_num_of_epochs}, batch: {i}/{len(list_of_observations) // 32}, Err_real - Err_fake = {critic_loss}, RL loss: {rl_loss}")
                else:
                    print(
                        f"Epoch: {epoch}/{max_num_of_epochs}, batch: {i}/{len(list_of_observations) // 32}, Err_real - Err_fake = {critic_loss}")

        if epoch % 500 == 0 and epoch != 0:
            #        if True:
            plot_graphs(rnn, output, absence_net, ECC_nodes, ECC_edges, epoch, device, args, qm9_smiles)
            save_model(ECC_nodes, ECC_edges, epoch)
