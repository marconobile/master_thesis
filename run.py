
from test import * 

VALIDATION, GENERATE = False, True

memorize_batch(max_epoch, rnn, output, train_dataset_loader, optimizer_rnn, optimizer_output, node_weights, edge_weights, scheduler_rnn, scheduler_output)

# while epoch <= max_epoch:
    # loss_this_epoch, loss_edg, loss_nodes = train_rnn_epoch(rnn=rnn, output=output,
    #                                                         data_loader_=train_dataset_loader,
    #                                                         optimizer_rnn=optimizer_rnn, optimizer_output=optimizer_output,
    #                                                         node_weights=node_weights, edge_weights=edge_weights)
    # scheduler_rnn.step()
    # scheduler_output.step()
    # if epoch % 100 == 0: train_log.info(f'Epoch: {epoch}/{max_epoch}, sum of Loss: {loss_this_epoch:.8f}, loss edges {loss_edg:.8f}, loss nodes {loss_nodes:.8f}')
    # if VALIDATION and epoch % 100 == 0:
    #     loss_this_epoch, loss_edg, loss_nodes = validate_rnn_epoch(rnn, output, val_dataset_loader, node_weights, edge_weights)
    #     val_log.info(f'Epoch: {epoch}/{max_epoch}, sum of Loss: {loss_this_epoch:.8f}, loss edges {loss_edg:.8f}, loss nodes {loss_nodes:.8f}')
    # epoch += 1

if GENERATE:
    Ns = [10]#, 60000, 110000, 160000, 210000]
    for i in Ns: generate_mols(i)

