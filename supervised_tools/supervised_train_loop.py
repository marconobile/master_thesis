import torch
import torch.nn.functional as F
import numpy as np
from args import Args


def train_rnn_epoch(rnn, output, data_loader_, optimizer_rnn, optimizer_output,
                    node_weights, edge_weights, device, absence_net, absence_net_opt):
    args = Args()

    rnn.train()
    output.train()
    absence_net.train()

    loss_sum_abs = 0
    loss_sum = 0
    loss_sum_edges = 0
    loss_sum_nodes = 0

    max_num_node = args.max_num_node
    max_prev_node = args.max_prev_node
    num_layers = rnn.num_layers
    edge_features_dim = args.edge_feature_dims

    for batch_idx, data in enumerate(data_loader_):

        rnn.zero_grad()
        output.zero_grad()
        absence_net.zero_grad()

        x_unsorted = data['x'].float()  # ([bs, max_num_node, max_prev_node, edge_feature])
        y_unsorted = data['y'].float()

        x_nodes_unsorted = data['x_node_attr'].float()  # ([bs, max_num_node, node_feature])
        y_nodes_unsorted = data['y_node_attr'].float()

        y_len_unsorted = data['len']  # list of seq_len of each g, len()=bs

        y_len_max = max(y_len_unsorted)  # pick max_seq_length of the current batch
        x_unsorted = x_unsorted[:, 0:y_len_max, :, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :, :]

        x_nodes_unsorted = x_nodes_unsorted[:, 0:y_len_max, :]
        y_nodes_unsorted = y_nodes_unsorted[:, 0:y_len_max, :]

        x_unsorted_for_nn = torch.reshape(x_unsorted, (
            x_unsorted.shape[0], x_unsorted.shape[1], x_unsorted.shape[2] * x_unsorted.shape[3]))
        y_unsorted_for_nn = torch.reshape(y_unsorted, (
            x_unsorted.shape[0], x_unsorted.shape[1], x_unsorted.shape[2] * x_unsorted.shape[3]))

        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted_for_nn.size(0))

        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted_for_nn, 0, sort_index)  # ([bs, max_seq_l, 8*4])
        y = torch.index_select(y_unsorted_for_nn, 0, sort_index)
        x_nodes = torch.index_select(x_nodes_unsorted, 0, sort_index)  # ([bs, max_seq_l, node_f])
        y_nodes = torch.index_select(y_nodes_unsorted, 0, sort_index)  # NODE TARGETS
        y_reshape = torch.nn.utils.rnn.pack_padded_sequence(y, y_len, batch_first=True).data
        idx = [i for i in range(y_reshape.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)  # inverts the rows order of y_reshape 
        y_reshape = y_reshape.view(y_reshape.size(0), max_prev_node, edge_features_dim)
        output_x = torch.cat((torch.ones(y_reshape.size(0), 1, edge_features_dim), y_reshape[:, 0:-1, :]), dim=1)
        output_y = y_reshape
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin) - 1, 0, -1):  # countdown from len(output_y_len_bin)
            count_temp = np.sum(output_y_len_bin[i:])  # count how many times y_len is above i
            if i == max_num_node:
                output_y_len.extend([min(max_prev_node, y.size(2))] * count_temp)
            else:
                output_y_len.extend([min(i, y.size(2))] * count_temp)

        x = torch.cat((x_nodes, x), 2)  # INPUT FOR NODE LVL
        x = torch.tensor(x).to(device)
        output_x = torch.tensor(output_x).to(device)

        y_nodes = torch.tensor(y_nodes).to(device)
        output_y = torch.tensor(output_y).to(device)

        # TARGETS
        output_sigmoid, output_y = torch.split(output_y, [1, edge_features_dim - 1], dim=2)  # new target

        # OUTPUTS
        h, node_prediction = rnn(x, pack=True, input_len=y_len)

        h = torch.nn.utils.rnn.pack_padded_sequence(h, y_len, batch_first=True).data  # get packed hidden vector
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = torch.tensor(torch.LongTensor(idx)).to(device)
        h = h.index_select(0, idx)

        hidden_null_1 = torch.tensor(torch.zeros(num_layers - 1, h.size(0), h.size(1))).to(device)
        hidden_null_2 = torch.tensor(torch.zeros(num_layers - 1, h.size(0), h.size(1))).to(device)

        output.hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null_1),
                                  dim=0)  # num_layers, batch_size, hidden_size

        absence_net.hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null_2),
                                       dim=0)  # num_layers, batch_size, hidden_size

        abs_ = absence_net(output_x, pack=True, input_len=output_y_len)

        y_pred = output(output_x, pack=True, input_len=output_y_len)

        #####################################################
        #############       OUTPUT LAYERS       #############
        #####################################################

        y_pred = F.log_softmax(y_pred, dim=2)
        node_prediction = F.log_softmax(node_prediction, dim=2)

        abs_ = F.sigmoid(abs_)

        abs_ = torch.nn.utils.rnn.pack_padded_sequence(abs_, output_y_len, batch_first=True)
        abs_ = torch.nn.utils.rnn.pad_packed_sequence(abs_, batch_first=True)[0]

        output_sigmoid = torch.nn.utils.rnn.pack_padded_sequence(output_sigmoid, output_y_len, batch_first=True)
        output_sigmoid = torch.nn.utils.rnn.pad_packed_sequence(output_sigmoid, batch_first=True)[0]

        # pack and pad predicted edges
        y_pred = torch.nn.utils.rnn.pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = torch.nn.utils.rnn.pad_packed_sequence(y_pred, batch_first=True)[0]

        # pack and pad real edges
        output_y = torch.nn.utils.rnn.pack_padded_sequence(output_y, output_y_len, batch_first=True)
        output_y = torch.nn.utils.rnn.pad_packed_sequence(output_y, batch_first=True)[0]

        # pack and pad predicted nodes
        node_prediction = torch.nn.utils.rnn.pack_padded_sequence(node_prediction, y_len, batch_first=True)
        node_prediction = torch.nn.utils.rnn.pad_packed_sequence(node_prediction, batch_first=True)[0]

        # pack and pad real nodes
        y_nodes = torch.nn.utils.rnn.pack_padded_sequence(y_nodes, y_len, batch_first=True)
        y_nodes = torch.nn.utils.rnn.pad_packed_sequence(y_nodes, batch_first=True)[0]

        abs_loss = F.binary_cross_entropy(abs_.to(device), output_sigmoid.to(device), reduction='mean',
                                          weight=edge_weights[0].to(torch.float32).to(device))

        # permutations of predicted edg/nodes
        y_pred = y_pred.permute(0, 2, 1)
        node_prediction = node_prediction.permute(0, 2, 1)

        # permutations of real edg/nodes
        output_y = output_y.permute(0, 2, 1)
        y_nodes = y_nodes.permute(0, 2, 1)

        values, indices_edges = torch.max(output_y, 1)

        values, indices_nodes = torch.max(y_nodes, 1)

        edge_loss = F.nll_loss(y_pred, indices_edges, reduction='mean',
                               weight=edge_weights[1:].to(torch.float32).to(device))

        node_loss = F.nll_loss(node_prediction, indices_nodes, reduction='mean',
                               weight=node_weights.to(torch.float32).to(device))

        loss = edge_loss + node_loss + abs_loss

        loss.backward(retain_graph=True)
        optimizer_output.step()
        optimizer_rnn.step()
        absence_net_opt.step()

        loss_sum += loss.data
        loss_sum_edges += edge_loss.data
        loss_sum_nodes += node_loss.data
        loss_sum_abs += abs_loss.data

    return loss_sum / (batch_idx + 1), loss_sum_edges / (batch_idx + 1), loss_sum_nodes / (
            batch_idx + 1), loss_sum_abs / (batch_idx + 1)


def test_rnn_single_epoch(rnn, output, data_loader_, node_weights, edge_weights, device, absence_net):
    args = Args()

    rnn.eval()
    output.eval()
    absence_net.eval()

    loss_sum_abs = 0
    loss_sum = 0
    loss_sum_edges = 0
    loss_sum_nodes = 0

    max_num_node = args.max_num_node
    max_prev_node = args.max_prev_node
    num_layers = rnn.num_layers
    edge_features_dim = args.edge_feature_dims

    for batch_idx, data in enumerate(data_loader_):

        rnn.zero_grad()
        output.zero_grad()
        absence_net.zero_grad()

        x_unsorted = data['x'].float()  # ([bs, max_num_node, max_prev_node, edge_feature])
        y_unsorted = data['y'].float()

        x_nodes_unsorted = data['x_node_attr'].float()  # ([bs, max_num_node, node_feature])
        y_nodes_unsorted = data['y_node_attr'].float()

        y_len_unsorted = data['len']  # list of seq_len of each g, len()=bs

        y_len_max = max(y_len_unsorted)  # pick max_seq_length of the current batch
        x_unsorted = x_unsorted[:, 0:y_len_max, :, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :, :]

        x_nodes_unsorted = x_nodes_unsorted[:, 0:y_len_max, :]
        y_nodes_unsorted = y_nodes_unsorted[:, 0:y_len_max, :]

        x_unsorted_for_nn = torch.reshape(x_unsorted, (
            x_unsorted.shape[0], x_unsorted.shape[1], x_unsorted.shape[2] * x_unsorted.shape[3]))
        y_unsorted_for_nn = torch.reshape(y_unsorted, (
            x_unsorted.shape[0], x_unsorted.shape[1], x_unsorted.shape[2] * x_unsorted.shape[3]))

        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted_for_nn.size(0))

        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted_for_nn, 0, sort_index)  # ([bs, max_seq_l, 8*4])
        y = torch.index_select(y_unsorted_for_nn, 0, sort_index)
        x_nodes = torch.index_select(x_nodes_unsorted, 0, sort_index)  # ([bs, max_seq_l, node_f])
        y_nodes = torch.index_select(y_nodes_unsorted, 0, sort_index)  # NODE TARGETS
        y_reshape = torch.nn.utils.rnn.pack_padded_sequence(y, y_len, batch_first=True).data
        idx = [i for i in range(y_reshape.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)  # inverts the rows order of y_reshape
        y_reshape = y_reshape.view(y_reshape.size(0), max_prev_node, edge_features_dim)
        output_x = torch.cat((torch.ones(y_reshape.size(0), 1, edge_features_dim), y_reshape[:, 0:-1, :]), dim=1)
        output_y = y_reshape
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin) - 1, 0, -1):  # countdown from len(output_y_len_bin)
            count_temp = np.sum(output_y_len_bin[i:])  # count how many times y_len is above i
            if i == max_num_node:
                output_y_len.extend([min(max_prev_node, y.size(2))] * count_temp)
            else:
                output_y_len.extend([min(i, y.size(2))] * count_temp)

        x = torch.cat((x_nodes, x), 2)  # INPUT FOR NODE LVL
        x = torch.tensor(x).to(device)
        output_x = torch.tensor(output_x).to(device)

        y_nodes = torch.tensor(y_nodes).to(device)
        output_y = torch.tensor(output_y).to(device)

        # TARGETS
        output_sigmoid, output_y = torch.split(output_y, [1, edge_features_dim - 1], dim=2)  # new target

        # OUTPUTS
        h, node_prediction = rnn(x, pack=True, input_len=y_len)

        h = torch.nn.utils.rnn.pack_padded_sequence(h, y_len, batch_first=True).data  # get packed hidden vector
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = torch.tensor(torch.LongTensor(idx)).to(device)
        h = h.index_select(0, idx)

        hidden_null_1 = torch.tensor(torch.zeros(num_layers - 1, h.size(0), h.size(1))).to(device)
        hidden_null_2 = torch.tensor(torch.zeros(num_layers - 1, h.size(0), h.size(1))).to(device)

        output.hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null_1),
                                  dim=0)  # num_layers, batch_size, hidden_size

        absence_net.hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null_2),
                                       dim=0)  # num_layers, batch_size, hidden_size

        abs_ = absence_net(output_x, pack=True, input_len=output_y_len)

        y_pred = output(output_x, pack=True, input_len=output_y_len)

        #####################################################
        #############       OUTPUT LAYERS       #############
        #####################################################

        y_pred = F.log_softmax(y_pred, dim=2)
        node_prediction = F.log_softmax(node_prediction, dim=2)

        abs_ = F.sigmoid(abs_)

        abs_ = torch.nn.utils.rnn.pack_padded_sequence(abs_, output_y_len, batch_first=True)
        abs_ = torch.nn.utils.rnn.pad_packed_sequence(abs_, batch_first=True)[0]

        output_sigmoid = torch.nn.utils.rnn.pack_padded_sequence(output_sigmoid, output_y_len, batch_first=True)
        output_sigmoid = torch.nn.utils.rnn.pad_packed_sequence(output_sigmoid, batch_first=True)[0]

        # pack and pad predicted edges
        y_pred = torch.nn.utils.rnn.pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = torch.nn.utils.rnn.pad_packed_sequence(y_pred, batch_first=True)[0]

        # pack and pad real edges
        output_y = torch.nn.utils.rnn.pack_padded_sequence(output_y, output_y_len, batch_first=True)
        output_y = torch.nn.utils.rnn.pad_packed_sequence(output_y, batch_first=True)[0]

        # pack and pad predicted nodes
        node_prediction = torch.nn.utils.rnn.pack_padded_sequence(node_prediction, y_len, batch_first=True)
        node_prediction = torch.nn.utils.rnn.pad_packed_sequence(node_prediction, batch_first=True)[0]

        # pack and pad real nodes
        y_nodes = torch.nn.utils.rnn.pack_padded_sequence(y_nodes, y_len, batch_first=True)
        y_nodes = torch.nn.utils.rnn.pad_packed_sequence(y_nodes, batch_first=True)[0]

        abs_loss = F.binary_cross_entropy(abs_.to(device), output_sigmoid.to(device), reduction='mean',
                                          weight=edge_weights[0].to(torch.float32).to(device))

        # permutations of predicted edg/nodes
        y_pred = y_pred.permute(0, 2, 1)
        node_prediction = node_prediction.permute(0, 2, 1)

        # permutations of real edg/nodes
        output_y = output_y.permute(0, 2, 1)
        y_nodes = y_nodes.permute(0, 2, 1)

        values, indices_edges = torch.max(output_y, 1)

        values, indices_nodes = torch.max(y_nodes, 1)

        edge_loss = F.nll_loss(y_pred, indices_edges, reduction='mean',
                               weight=edge_weights[1:].to(torch.float32).to(device))

        node_loss = F.nll_loss(node_prediction, indices_nodes, reduction='mean',
                               weight=node_weights.to(torch.float32).to(device))

        # no backward step
        loss = edge_loss + node_loss + abs_loss
        loss_sum += loss.data
        loss_sum_edges += edge_loss.data
        loss_sum_nodes += node_loss.data
        loss_sum_abs += abs_loss.data

    return loss_sum / (batch_idx + 1), loss_sum_edges / (batch_idx + 1), loss_sum_nodes / (
            batch_idx + 1), loss_sum_abs / (batch_idx + 1)
