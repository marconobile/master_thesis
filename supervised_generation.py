from create_train_val_data import create_train_val_dataloaders, get_log_weights, get_sklearn_weights
# from model import GRU_plain
import torch
# from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
from data_utils import mols_smiles_plots, mols_txt
from save_load_model import save
from train import train_rnn_epoch, test_rnn_single_epoch
from new_gen import new_gen
from get_generator import get_generator
from args import Args
from no_chains import plot_graphs_no_chains

def supervised_generation(dataset, with_absence_net, device, cuda, train_log, test_log, qm9_smiles):
    args = Args()
    max_num_node = args.max_num_node
    max_prev_node = args.max_prev_node

    train_dataset_loader, test_dataset_loader, for_weights = create_train_val_dataloaders(dataset, max_num_node,
                                                                                          max_prev_node)

    LR = 1e-3

    # get weights for NLL
    node_weights, edge_weights = get_log_weights(for_weights,args )
    #node_weights, edge_weights = get_sklearn_weights(for_weights, max_num_node, max_prev_node)

    print("#N. batches in train_dataset_loader: ", len(train_dataset_loader), "#N. batches in test_dataset_loader: ",
          len(test_dataset_loader))
    # print('node_weights', node_weights, '\nedge_weights', edge_weights)

    #node_weights = torch.ones_like(node_weights).to(device)
    #edge_weights = torch.ones_like(edge_weights).to(device)
    edge_weights[0]=100
    print('node_weights', node_weights, '\nedge_weights', edge_weights)

    # get networks:
    if with_absence_net:
        rnn, output, absence_net = get_generator()
        optimizer_abs_net = torch.optim.Adam(list(absence_net.parameters()), lr=LR, weight_decay=5e-5)
        scheduler_abs = False # torch.optim.lr_scheduler.MultiStepLR(optimizer_abs_net, milestones=[1000], gamma=.1)
    else:
        rnn, output, _ = get_generator()

    print('Networks skeletons:')
    print(rnn)
    print('##' * 20)
    print(output)
    if absence_net:
        print('##' * 20)
        print(absence_net)

    if cuda:
        rnn.to(device)
        output.to(device)
        if absence_net:
            absence_net.to(device)

    print('Nets structures loaded correctly!')
    print('Launching: tensorboard and training!')
    # now = datetime.now()
    # tsb_dir = f'./runs/run_{str(now)}'
    # writer = SummaryWriter(log_dir=tsb_dir)

    optimizer_rnn = torch.optim.Adam(list(rnn.parameters()), lr=LR, weight_decay=5e-5)
    optimizer_output = torch.optim.Adam(list(output.parameters()), lr=LR, weight_decay=5e-5)

    # optimizer_rnn = torch.optim.RMSprop(list(rnn.parameters()), lr=LR)
    # optimizer_output = torch.optim.RMSprop(list(output.parameters()), lr=LR)

    scheduler_rnn = False  # torch.optim.lr_scheduler.MultiStepLR(optimizer_rnn, milestones=[192000], gamma=.1)
    scheduler_output = False  # torch.optim.lr_scheduler.MultiStepLR(optimizer_output, milestones=[192000],
    # gamma=.1)

    epoch = 1
    max_epoch = 3500
    counter_test = 0

    # train_loss_list = []
    # test_loss_list = []
    patience = 100

    while epoch <= max_epoch:

        if not with_absence_net:
            loss_this_epoch, loss_edg, loss_nodes = train_rnn_epoch(epoch=epoch, rnn=rnn, output=output,
                                                                    data_loader_=train_dataset_loader,
                                                                    optimizer_rnn=optimizer_rnn,
                                                                    optimizer_output=optimizer_output,
                                                                    scheduler_rnn=scheduler_rnn,
                                                                    scheduler_output=scheduler_output,
                                                                    node_weights=node_weights,
                                                                    edge_weights=edge_weights,
                                                                    device=device,
                                                                    absence_net=False,
                                                                    optimizer_abs_net=False,
                                                                    scheduler_abs=False
                                                                    )

            train_log.info(
                f'Epoch: {epoch}/{max_epoch}, sum of Loss: {loss_this_epoch:.8f}, loss edges {loss_edg:.8f}, loss nodes {loss_nodes:.8f}')

        else:
            loss_this_epoch, loss_edg, loss_nodes, loss_abs = train_rnn_epoch(epoch=epoch, rnn=rnn, output=output,
                                                                              data_loader_=train_dataset_loader,
                                                                              optimizer_rnn=optimizer_rnn,
                                                                              optimizer_output=optimizer_output,
                                                                              scheduler_rnn=scheduler_rnn,
                                                                              scheduler_output=scheduler_output,
                                                                              node_weights=node_weights,
                                                                              edge_weights=edge_weights,
                                                                              device=device,
                                                                              absence_net=absence_net,
                                                                              absence_net_opt=optimizer_abs_net,
                                                                              scheduler_abs=scheduler_abs
                                                                              )

            train_log.info(
                f'Epoch: {epoch}/{max_epoch}, sum of Loss: {loss_this_epoch:.8f}, loss edges {loss_edg:.8f}, loss nodes {loss_nodes:.8f} , loss_abs {loss_abs:.8f}')
            #print("Epoch: ", epoch, "optimizer_abs_net initial_lr =  ", optimizer_abs_net.param_groups[0]['initial_lr'],
            #      " current lr = ",
            #      optimizer_abs_net.param_groups[0]['lr'])

        #if epoch % patience == 0:
        if False:
            if not with_absence_net:
                test_loss, loss_edg_test, loss_nodes_test = test_rnn_single_epoch(epoch=epoch, rnn=rnn, output=output,
                                                                                  data_loader_=test_dataset_loader,
                                                                                  node_weights=node_weights,
                                                                                  edge_weights=edge_weights,
                                                                                  device=device,
                                                                                  absence_net=absence_net)
                test_log.info(
                    f'Evaluation step number {counter_test + 1} (epoch {epoch}), total loss value: {test_loss:.8f}, loss edges {loss_edg_test:.8f}, loss nodes {loss_nodes_test:.8f} ')
                counter_test += 1
            else:
                test_loss, loss_edg_test, loss_nodes_test, loss_abs = test_rnn_single_epoch(epoch=epoch, rnn=rnn,
                                                                                            output=output,
                                                                                            data_loader_=test_dataset_loader,
                                                                                            node_weights=node_weights,
                                                                                            edge_weights=edge_weights,
                                                                                            device=device,
                                                                                            absence_net=absence_net)
                test_log.info(
                    f'Evaluation step number {counter_test + 1} (epoch {epoch}), total loss value: {test_loss:.8f}, loss edges {loss_edg_test:.8f}, loss nodes {loss_nodes_test:.8f} , loss_abs {loss_abs:.8f}')
                counter_test += 1

        if epoch % patience == 0:  # and not stop_training:
            save(epoch=epoch, rnn=rnn, output=output, absence_net=absence_net)
            print(f'Model saved at epoch {epoch}!')

        if epoch % patience == 0:
        #if True:
            rnn.eval()
            output.eval()
            if absence_net:
                absence_net.eval()
            n_of_graph_to_be_generated = 6400
            print(f'Generating {n_of_graph_to_be_generated} graphs for epoch', epoch)

            to_draw = []
            for jj in range(n_of_graph_to_be_generated):
                obs = new_gen(rnn, output, absence_net, device, args=args, test_batch_size=1, max_num_node=max_num_node,
                              max_prev_node=max_prev_node)
                to_draw.append(obs)

            rdkit_mols, mols_smiles = mols_smiles_plots(to_draw, './figures/FIG_epoch_' + str(epoch), epoch)
            mols_txt(epoch, rdkit_mols, mols_smiles, qm9_smiles)

            plot_graphs_no_chains(rnn, output, absence_net, epoch, device, args, qm9_smiles)

        epoch += 1

    # writer.close()

    save(epoch=epoch, rnn=rnn, output=output, absence_net=absence_net)

    print(f'Model saved after LAST epoch {epoch}!')
    print('Writer closed, networks trained')
    print('Script END')
