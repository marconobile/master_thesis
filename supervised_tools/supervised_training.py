from supervised_tools.create_train_val_data import create_train_val_dataloaders, get_log_weights
import torch
from utils.data_utils import mols_smiles_plots, mols_txt
from supervised_tools.save_load_model import save
from supervised_tools.supervised_train_loop import train_rnn_epoch, test_rnn_single_epoch
from supervised_tools.generate_single_obs import generate_single_obs
from supervised_tools.supervised_model import get_generator
from args import Args
import pickle


def supervised_training(dataset, device, cuda, train_log, test_log, qm9_smiles, test_set):

    '''
    dataset: list of pyg Data objects
    '''


    print("Starting supervised training...")
    args = Args()
    max_num_node = args.max_num_node
    max_prev_node = args.max_prev_node
    LR = 5e-4

    train_dataset_loader, _ = create_train_val_dataloaders(dataset, max_num_node,
                                                           max_prev_node)

    if args.test_set:
        test_dataset_loader, _ = create_train_val_dataloaders(test_set, max_num_node,
                                                              max_prev_node)

    # get weights for NLL
    node_weights, edge_weights = get_log_weights(dataset, args)
    edge_weights[0] = 100  # ad-hoc choice

    print("#N. batches in train_dataset_loader: ", len(train_dataset_loader))
    print('Node_weights', node_weights, '\nEdge_weights', edge_weights)

    # get models:
    rnn, output, absence_net = get_generator()
    optimizer_abs_net = torch.optim.Adam(list(absence_net.parameters()), lr=LR, weight_decay=5e-5)
    optimizer_rnn = torch.optim.Adam(list(rnn.parameters()), lr=LR, weight_decay=5e-5)
    optimizer_output = torch.optim.Adam(list(output.parameters()), lr=LR, weight_decay=5e-5)

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
    max_epoch = 3001
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

        if args.test_set and epoch % patience == 0:
            test_loss, loss_edg_test, loss_nodes_test, loss_abs = test_rnn_single_epoch(rnn=rnn,
                                                                                        output=output,
                                                                                        data_loader_=test_dataset_loader,
                                                                                        node_weights=node_weights,
                                                                                        edge_weights=edge_weights,
                                                                                        device=device,
                                                                                        absence_net=absence_net)
            test_log.info(
                f'Evaluation step number {counter_test + 1} (epoch {epoch}), total loss value: {test_loss:.8f}, loss edges {loss_edg_test:.8f}, loss nodes {loss_nodes_test:.8f} , loss_abs {loss_abs:.8f}')
            counter_test += 1

        if epoch % patience == 0:
            save(epoch=epoch, rnn=rnn, output=output, absence_net=absence_net)
            print(f'Model saved at epoch {epoch}!')

            rnn.eval()
            output.eval()
            absence_net.eval()

            n_of_graph_to_be_generated = 1600
            print(f'Generating {n_of_graph_to_be_generated} graphs for epoch', epoch)

            to_draw = []
            for jj in range(n_of_graph_to_be_generated):
                obs = generate_single_obs(rnn, output, absence_net, device, args=args, test_batch_size=1,
                                          max_num_node=max_num_node,
                                          max_prev_node=max_prev_node)
                to_draw.append(obs)

            rdkit_mols, mols_smiles = mols_smiles_plots(to_draw, './figures/FIG_epoch_' + str(epoch))
            mols_txt(epoch, rdkit_mols, mols_smiles, qm9_smiles)

        epoch += 1

    save(epoch=epoch, rnn=rnn, output=output, absence_net=absence_net)

    print(f'Model saved after LAST epoch {epoch}!')
    print('Writer closed, networks trained')
    print('Script END')
    return None
