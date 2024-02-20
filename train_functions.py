import torch.nn.init as init
from mappings import *
from utils.data_utils import pyg2rdkit, save_smiles
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
from rdkit import Chem
import torch
print(torch.__version__)

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias != None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def train_rnn_epoch(rnn, output, data_loader_, optimizer, node_weights, edge_weights, scheduler=None):
    rnn.train()
    output.train()
    tot_loss, loss_sum_edges, loss_sum_nodes = 0, 0, 0
    for batch_idx, data in enumerate(data_loader_):
        rnn.zero_grad()
        output.zero_grad()
        loss, edge_loss, node_loss = fit_batch(data, rnn, output, node_weights, edge_weights)
        loss.backward()
        # nn.utils.clip_grad_value_(list(rnn.parameters()) + list(output.parameters()), clip_value=.25)
        optimizer.step()        
        if scheduler != None: scheduler.step()
        loss_sum_edges += edge_loss.data
        loss_sum_nodes += node_loss.data
        tot_loss =  loss_sum_edges + loss_sum_nodes
    return tot_loss / (batch_idx + 1), loss_sum_edges / (batch_idx + 1), loss_sum_nodes / (batch_idx + 1)


@torch.no_grad()
def validate_rnn_epoch(rnn, output, data_loader_, node_weights, edge_weights):
    rnn.eval()
    output.eval()    
    loss_sum_edges, loss_sum_nodes = 0, 0
    for batch_idx, data in enumerate(data_loader_):
        loss, edge_loss, node_loss = fit_batch(data, rnn, output, node_weights, edge_weights)
        loss_sum_edges += edge_loss.data
        loss_sum_nodes += node_loss.data
    return loss / (batch_idx + 1), loss_sum_edges / (batch_idx + 1), loss_sum_nodes / (batch_idx + 1)


def fit_batch(data, rnn, output, node_weights, edge_weights):

    for k, v in data.items(): data[k] = v.to(device)

    ce_nodes = torch.nn.CrossEntropyLoss(node_weights)
    ce_edges = torch.nn.CrossEntropyLoss(edge_weights)

    with torch.no_grad():
        # ([bs, max_num_node, max_prev_node, edge_feature])
        y_unsorted = data['y'].float()
        x_unsorted = data['x'].float()

        # ([bs, max_num_node, node_feature])
        y_nodes_unsorted = data['y_node_attr'].float()
        x_nodes_unsorted = data['x_node_attr'].float()

        y_len_unsorted = data['len']  # list of seq_len of each g, len()=bs
        y_len_max = max(y_len_unsorted)  # pick max_seq_length of the current batch
        x_unsorted = x_unsorted[:, :y_len_max, :, :]
        y_unsorted = y_unsorted[:, :y_len_max, :, :]

        x_nodes_unsorted = x_nodes_unsorted[:, :y_len_max, :]
        y_nodes_unsorted = y_nodes_unsorted[:, :y_len_max, :]

        x_unsorted_for_nn = torch.reshape(x_unsorted, (x_unsorted.shape[0], x_unsorted.shape[1], x_unsorted.shape[2] * x_unsorted.shape[3]))
        y_unsorted_for_nn = torch.reshape(y_unsorted, (x_unsorted.shape[0], x_unsorted.shape[1], x_unsorted.shape[2] * x_unsorted.shape[3]))

        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.cpu() # ([bs, max_seq_l, 8*4])
        x = torch.index_select(x_unsorted_for_nn, 0, sort_index)
        y = torch.index_select(y_unsorted_for_nn, 0, sort_index) # ([bs, max_seq_l, node_f])
        x_nodes = torch.index_select(x_nodes_unsorted, 0, sort_index)
        y_nodes = torch.index_select(y_nodes_unsorted, 0, sort_index) # NODE TARGETS
        y_reshape = torch.nn.utils.rnn.pack_padded_sequence(y, y_len, batch_first=True).data
        idx = torch.tensor([i for i in range(y_reshape.size(0) - 1, -1, -1)], device=device, dtype=torch.long)
        # inverts the rows order of y_reshape
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0), max_prev_node, edge_feature_dims)
        output_x = torch.cat((torch.ones(y_reshape.size(0), 1, edge_feature_dims, device=device), y_reshape[:, 0:-1, :]), dim=1)
        output_y = y_reshape
        output_y_len = []
        output_y_len_bin = torch.bincount(y_len)
        for i in range(len(output_y_len_bin) - 1, 0, -1):  # countdown from len(output_y_len_bin)
            count_temp = torch.sum(output_y_len_bin[i:]) # count how many times y_len is above i
            if i == max_num_node: output_y_len.extend([min(max_prev_node, y.size(2))] * count_temp)
            else: output_y_len.extend([min(i, y.size(2))] * count_temp)
        x = torch.cat((x_nodes, x), 2) # INPUT FOR NODE LVL

    # OUTPUTS
    rnn.hidden = rnn.init_hidden(batch_size=x_unsorted_for_nn.size(0))
    h, node_prediction = rnn(x, pack=True, input_len=y_len)

    h = torch.nn.utils.rnn.pack_padded_sequence(h, y_len, batch_first=True).data  # get packed hidden vector
    idx = torch.tensor([i for i in range(h.size(0) - 1, -1, -1)], device=device, dtype=torch.long)
    h = h.index_select(0, idx)

    hidden_null = torch.zeros(rnn.num_layers - 1, h.size(0), h.size(1), device=device) # num_layers, batch_size, hidden_size
    output.hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null), dim=0)
    y_pred = output(output_x, pack=True, input_len=output_y_len)

    # OUTPUT LAYERS    
    y_pred = torch.nn.utils.rnn.pack_padded_sequence(y_pred, output_y_len, batch_first=True)    
    output_y = torch.nn.utils.rnn.pack_padded_sequence(output_y, output_y_len, batch_first=True)

    node_prediction = torch.nn.utils.rnn.pack_padded_sequence(node_prediction, y_len, batch_first=True)
    y_nodes = torch.nn.utils.rnn.pack_padded_sequence(y_nodes, y_len, batch_first=True)

    # edge_loss = F.cross_entropy(y_pred.data, output_y.data, weight = edge_weights)
    # node_loss = F.cross_entropy(node_prediction.data, y_nodes.data, weight = node_weights)

    edge_loss = ce_edges(y_pred[0], output_y[0])
    node_loss = ce_nodes(node_prediction[0], y_nodes[0])

    return edge_loss + node_loss, edge_loss, node_loss


@torch.no_grad()
def generate_single_obs(rnn, output, device, max_num_node, max_prev_node, test_batch_size=1):
    rnn.eval()
    output.eval()
    rnn.hidden = rnn.init_hidden(test_batch_size) #! rand    
    x_step = torch.ones((test_batch_size, 1, max_prev_node * edge_feature_dims + node_feature_dims), device=device)
    x_list, edg_attr_list, edg_idx_list = [], [], [] # initialize empty lists for Data() object
    
    for i in range(max_num_node): # Node RNN for-loop
        h, node_prediction = rnn(x_step)
        node_prediction_argmax = F.one_hot(node_prediction.argmax(), num_classes=node_feature_dims)
        x_list.append(node_prediction_argmax)

        # reset and update input for next iteration
        x_step = torch.zeros((test_batch_size, 1, max_prev_node * edge_feature_dims + node_feature_dims), device=device)
        x_step[:, :, :node_feature_dims] = node_prediction_argmax.data

        # init Edge/Abs lvl
        hidden_null = torch.zeros((rnn.num_layers - 1, h.size(0), h.size(2))).to(device)
        output.hidden = torch.cat((h.permute(1, 0, 2), hidden_null), dim=0).to(device)

        # token Edge lvl - randn best result
        output_x_step = torch.ones(test_batch_size, 1, edge_feature_dims).to(device)
        edge_rnn_step = 0
        idx = [k for k in range(i, -1, -1)] # this list is used to create edg_idx
        for j in range(min(max_prev_node, i + 1)): # Edge/Abs RNN for-loop          
            output_y_pred_step_out = output(output_x_step) # prediction for each and every prev node
            output.hidden = output.hidden.data.to(device)
            output_x_step_argmax = F.one_hot(output_y_pred_step_out.argmax(), num_classes=edge_feature_dims)       
     
            if torch.argmax(output_x_step_argmax, dim=-1) != 0:
                if i + 1 <= max_prev_node:                    
                    idx_select = torch.tensor([1, 2, 3, 4], dtype=torch.long).to(device) # select [1:]
                    edge_to_append = torch.index_select(output_x_step_argmax, dim=-1, index=idx_select).to(device)

                    # Duplicate
                    edges_to_append_doubled = torch.cat((edge_to_append.squeeze(), edge_to_append.squeeze()), dim=0).to(device)

                    # reshape to (2x4)
                    edges_to_append_resh = torch.reshape(edges_to_append_doubled, (2, 4)).to(device)

                    # Append to edg_attr_list
                    edg_attr_list.append(edges_to_append_resh)

                    # Edge_index creation
                    edg_idx_list.append(torch.tensor([i + 1, idx[j]]))
                    edg_idx_list.append(torch.tensor([idx[j], i + 1]))

            # Define next time-step input
            x_step[:, :, 4 * j + node_feature_dims + j: 4 * (j + 1) + node_feature_dims + (j + 1)] = output_x_step_argmax.data
            edge_rnn_step = j

        _, edges_to_break = torch.split(x_step, [node_feature_dims, max_prev_node * edge_feature_dims], dim=2)
        edges_to_break_temp = torch.reshape(edges_to_break, (edges_to_break.shape[0], max_prev_node, 5)).to(device)
        edges_to_break_uptillnow = edges_to_break_temp[0,:edge_rnn_step + 1, :].to(device)
        break_ = True
        for row in edges_to_break_uptillnow:
            if torch.argmax(row).item() != 0:
                break_ = False

        if break_:
            break

    # Stacking lists as tensors
    x_temp = torch.stack(x_list).to(device)

    # In both cases we need to reshape to (N,4)
    x = torch.reshape(x_temp, (x_temp.shape[0], x_temp.shape[-1])).to(device)

    if len(edg_idx_list) != 0:
        edge_idx_temp = torch.stack(edg_idx_list).to(device)
        edge_idx = torch.transpose(edge_idx_temp, 0, 1).to(device)

        # Stack on edge_attributes
        edge_attr_ = torch.stack(edg_attr_list, dim=0).to(device)

        # Reshape to (2*E,4)
        edge_attr = torch.reshape(edge_attr_, (edge_attr_.shape[0] * edge_attr_.shape[1], edge_attr_.shape[-1])).to(device)
    else:
        # print('disconnected nodes case')
        return Data(x=x.to(torch.float32).to(device))
    return Data(x=x.to(torch.float32), edge_index=edge_idx.to(torch.long),edge_attr=edge_attr.to(torch.float32)).to(device)


@torch.no_grad()
def generate_mols(N, rnn, output, epoch):
    to_draw = []
    for idx in range(N):
        print(f"{idx+1}/{N}", end='\r')
        obs = generate_single_obs(rnn, output, device, test_batch_size=1,
                                    max_num_node=max_num_node,
                                    max_prev_node=max_prev_node)
        to_draw.append(obs)

    smiles_ = pyg2rdkit(to_draw)
    filename = f"original_thesis_weights_all_{epoch}_{N}.smiles"
    smiles_ = [Chem.MolToSmiles(m) for m in smiles_]
    # smiles_ = []
    # try: 
    #     for m in smiles_:
    #         smiles_.append(Chem.MolToSmiles(m))
    # except:
    #     pass

    save_smiles(smiles_, ".", filename, "smiles")


def memorize_batch_single_opt(max_epoch, rnn, output, data_loader_, optimizer, node_weights, edge_weights, scheduler=None):
    rnn.train()
    output.train()    
    epoch = 1
    for _, data in enumerate(data_loader_): data = data
    while epoch <= max_epoch:
        rnn.zero_grad()
        output.zero_grad()
        loss, edge_loss, node_loss = fit_batch(data, rnn, output, node_weights, edge_weights)
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        if epoch % 100 == 0 or epoch == 1: 
            print(f'Epoch: {epoch}/{max_epoch}, lossEdges {edge_loss:.8f}, lossNodes {node_loss:.8f}')
        epoch += 1
    for i in [10]: generate_mols(i,rnn, output, epoch)

