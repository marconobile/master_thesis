import pickle
import torch
from copy import deepcopy
from supervised_tools.generate_single_obs import generate_single_obs
from utils.data_utils import mols_smiles_plots, mols_txt
from unsupervised_tools.discriminator_model import Critic
from unsupervised_tools.LineGraph_transform import LineGraph
from unsupervised_tools.gan_generator import Graph_ECC
from supervised_tools.save_load_model import load_nets
from utils.setup import unsupervised_setup
from unsupervised_tools.unsupervised_training_loop import train_plain_gan
from utils.data_utils import get_smiles
from rdkit import Chem
from unsupervised_tools.unsupervised_utils import filter_obs_by_ring_size


def unsupervised_training(dataset, smiles_list, cuda, device, args):
    print("Starting unsupervised training...")

    critic_loss_log, generator_loss_log = unsupervised_setup()
    epoch_to_load = args.graphRNN_epoch
    rnn, output, absence_net = load_nets(cuda, device, epoch_to_load)  # models in .eval()
    print(f'GraphRNN at epoch {epoch_to_load} loaded')

    if args.generate_train_set:
        number_of_observations = len(dataset)
        print(f'Starting generation: generating {number_of_observations} mols...')
        list_for_transf = []
        list_of_observations = []  # list of pyg obs
        while (len(list_of_observations) != number_of_observations):
            obs = generate_single_obs(rnn, output, absence_net, device, args=args, test_batch_size=1,
                                      max_num_node=args.max_num_node,
                                      max_prev_node=args.max_prev_node)

            obs_smile = get_smiles([obs])
            if obs_smile:
                m = Chem.MolFromSmiles(obs_smile[0])
                filter = filter_obs_by_ring_size(m)
            if m and filter:
                obs = obs.coalesce()
                list_of_observations.append(obs)
                obs_1 = deepcopy(obs)
                list_for_transf.append(obs_1)
        print('generation ended, plotting...')

        # saving to file
        with open('gan_dataset_generated_via_GraphRNN', 'wb') as fp:
            pickle.dump(list_of_observations, fp)

    # load from file
    print('Loading mols from OUTfile:')
    with open('gan_dataset_generated_via_GraphRNN', 'rb') as fp:
        list_of_observations = pickle.load(fp)

    list_for_transf = []

    for obs in list_of_observations:
        obs = obs.coalesce()
        obs_1 = deepcopy(obs)
        list_for_transf.append(obs_1)

    line_graphs_for_list_of_observations = []
    transf = LineGraph()
    for g in list_for_transf:
        line_graphs_for_list_of_observations.append(transf(g))

    rdkit_mols, mols_smiles = mols_smiles_plots(list_of_observations, './figures/gan_dataset_generated_via_GraphRNN')
    mols_txt(-1, rdkit_mols, mols_smiles, smiles_list)
    print('Generated dataset plotted')

    ECC_nodes = Graph_ECC(node_dims=args.node_feature_dims, edge_dims=args.edge_feature_dims - 1)
    ECC_edges = Graph_ECC(node_dims=args.edge_feature_dims - 1, edge_dims=args.node_feature_dims)
    critic = Critic(args=args)

    LR = 1e-5
    optimizer_nodes = torch.optim.RMSprop(list(ECC_nodes.parameters()), lr=LR)
    optimizer_edges = torch.optim.RMSprop(list(ECC_edges.parameters()), lr=LR)
    optimizer_critic = torch.optim.RMSprop(list(critic.parameters()), lr=LR)

    print('Starting gan training')
    if cuda:
        critic.to(device)
        ECC_nodes.to(device)
        ECC_edges.to(device)

    max_num_of_epochs = 5000
    N_CRITIC_STEPS = 10
    batch_size = 32

    train_plain_gan(max_num_of_epochs, batch_size, list_of_observations, line_graphs_for_list_of_observations,
                    dataset, device, N_CRITIC_STEPS, critic, ECC_nodes, ECC_edges, optimizer_critic,
                    optimizer_nodes, optimizer_edges, critic_loss_log, generator_loss_log,
                    rnn, output, absence_net, args, smiles_list)
