
from main import pyg2rdkit
from supervised_tools.create_train_val_data import create_train_val_dataloaders, get_log_weights
import torch

from utils.data_utils import mols_smiles_plots, mols_txt
from supervised_tools.save_load_model import save
from supervised_tools.supervised_train_loop import train_rnn_epoch, test_rnn_single_epoch
from supervised_tools.generate_single_obs import generate_single_obs
from supervised_tools.supervised_model import get_generator
# from new_model import get_generator
# import pickle


def supervised_training(dataset, device, cuda, train_log, test_log, qm9_smiles, test_set):
    '''
    dataset: list of pyg Data objects
    '''

    print("Starting supervised training...")
    
    max_num_node = 88
    max_prev_node = max_num_node - 1
    LR = 5e-2
    wd = 5e-3

    train_dataset_loader, _ = create_train_val_dataloaders(dataset, max_num_node,
                                                           max_prev_node)

    if test_set:
        test_dataset_loader, _ = create_train_val_dataloaders(test_set, max_num_node,
                                                              max_prev_node)

    # get weights for NLL
    # node_weights, edge_weights = get_log_weights(dataset, args)
    # edge_weights[0] = 100  # ad-hoc choice

    node_feature_dims = 12
    edge_feature_dims = 5
    node_weights = torch.ones((node_feature_dims))
    edge_weights = torch.ones((edge_feature_dims))

    print("#N. batches in train_dataset_loader: ", len(train_dataset_loader))
    # print('Node_weights', node_weights, '\nEdge_weights', edge_weights)

    # get models:
    rnn, output, absence_net = get_generator()
    optimizer_abs_net = torch.optim.RMSprop(list(absence_net.parameters()), lr=LR)# , weight_decay=wd)
    optimizer_rnn = torch.optim.RMSprop(list(rnn.parameters()), lr=LR)# , weight_decay=wd)
    optimizer_output = torch.optim.RMSprop(list(output.parameters()), lr=LR)# , weight_decay=wd)

    print('Networks skeletons:')
    print(rnn)
    print('##' * 20)
    print(output)
    print('##' * 20)
    print(absence_net)

    if cuda:
        rnn.to(device)
        output.to(device)
        absence_net.to(device)

    print('Nets structures loaded correctly! Training... ')

    epoch = 1  # starting epoch
    max_epoch = 10
    patience = 100
    counter_test = 0

    while epoch <= max_epoch:

        loss_this_epoch, loss_edg, loss_nodes, loss_abs = train_rnn_epoch(rnn=rnn, output=output,
                                                                          data_loader_=train_dataset_loader,
                                                                          optimizer_rnn=optimizer_rnn,
                                                                          optimizer_output=optimizer_output,
                                                                          node_weights=node_weights,
                                                                          edge_weights=edge_weights,
                                                                          device=device,
                                                                          absence_net=absence_net,
                                                                          absence_net_opt=optimizer_abs_net)

        train_log.info(
            f'Epoch: {epoch}/{max_epoch}, sum of Loss: {loss_this_epoch:.8f}, loss edges {loss_edg:.8f}, loss nodes {loss_nodes:.8f} , loss_abs {loss_abs:.8f}')

        # if test_set and epoch % patience == 0:
        #     test_loss, loss_edg_test, loss_nodes_test, loss_abs = test_rnn_single_epoch(rnn=rnn,
        #                                                                                 output=output,
        #                                                                                 data_loader_=test_dataset_loader,
        #                                                                                 node_weights=node_weights,
        #                                                                                 edge_weights=edge_weights,
        #                                                                                 device=device,
        #                                                                                 absence_net=absence_net)
        #     test_log.info(
        #         f'Evaluation step number {counter_test + 1} (epoch {epoch}), total loss value: {test_loss:.8f}, loss edges {loss_edg_test:.8f}, loss nodes {loss_nodes_test:.8f} , loss_abs {loss_abs:.8f}')
        #     counter_test += 1

    #     if epoch % patience == 0:
    #         save(epoch=epoch, rnn=rnn, output=output, absence_net=absence_net)
    #         print(f'Model saved at epoch {epoch}!')

    #         rnn.eval()
    #         output.eval()
    #         absence_net.eval()

    #         n_of_graph_to_be_generated = 1600
    #         print(f'Generating {n_of_graph_to_be_generated} graphs for epoch', epoch)

    #         to_draw = []
    #         for _ in range(n_of_graph_to_be_generated):
    #             obs = generate_single_obs(rnn, output, absence_net, device, test_batch_size=1,
    #                                       max_num_node=max_num_node,
    #                                       max_prev_node=max_prev_node)
    #             to_draw.append(obs)

    #         rdkit_mols, mols_smiles = mols_smiles_plots(to_draw, './figures/FIG_epoch_' + str(epoch))
    #         mols_txt(epoch, rdkit_mols, mols_smiles, qm9_smiles)

        epoch += 1

    # save(epoch=epoch, rnn=rnn, output=output, absence_net=absence_net)

    print(f'Model saved after LAST epoch {epoch}!')
    print('Writer closed, networks trained')
    print('Script END')
    # return None

    # ------------------------------------------------------------------------------------------


    def save_smiles(smiles, path, filename, ext='.txt'):
        import os
        '''
        saves smiles in a file at path
        extension can be provided in filename or as separate arg
        args:
            - smiles str iterable 
            - path directory where to save smiles 
            - filename name of the file, must not have extension
        '''
        path_to_file = os.path.join(path, filename)
        filename_ext = os.path.splitext(path_to_file)[-1].lower()
        if not filename_ext:
            if ext not in ['.txt', '.smiles']:
                raise f"extension {ext} not valid"
            path_to_file += ext

        # path_to_file = generate_file(path, filename)
        with open(path_to_file, "w+") as f:
            f.writelines("%s\n" % smi for smi in smiles)

    Ns = [5]#, 60000, 110000, 160000, 210000]

    # for each el in list, call f with el
    # from utils.data_utils import get_smiles

    def generate_mols(N):
        to_draw = []
        for idx in range(N):
            print(f"{idx+1}/{N}", end='\r')
            obs = generate_single_obs(rnn, output, device, test_batch_size=1,
                                        max_num_node=max_num_node,
                                        max_prev_node=max_prev_node)
            to_draw.append(obs)        

        smiles_ = pyg2rdkit(to_draw)
        # path = "/home/nobilm@usi.ch/wd/data/generated_smiles/graphRNN_original_thesis_weights/"
        from rdkit import Chem
        filename = f"original_thesis_weights_all_{epoch}_{N}.smiles"
        smiles_ = [Chem.MolToSmiles(m) for m in smiles_]
        save_smiles(smiles_, ".", filename, "smiles")


    for i in Ns: generate_mols(i)
