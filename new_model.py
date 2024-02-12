import torch
import torch.nn.functional as F
import numpy as np

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")

node_feature_dims = 12
edge_feature_dims = 5
max_num_node = 88
max_prev_node = max_num_node - 1


def get_generator():
    # NODE LEVEL AND ABSENCE NET embeddings and hidden sizes
    embedding_size_rnn = 256
    hidden_size_rnn = 256

    # EDGE LEVEL embeddings and hidden sizes
    embedding_size_rnn_output = 128
    hidden_size_rnn_output = 128
    out_edge_level = hidden_size_rnn_output

    rnn = GRU_plain(input_size=node_feature_dims + edge_feature_dims * max_prev_node,
                    embedding_size=embedding_size_rnn,
                    hidden_size=hidden_size_rnn,
                    output_size=hidden_size_rnn_output,
                    node_lvl=True,
                    out_middle_layer=hidden_size_rnn_output)

    output = GRU_plain(input_size=edge_feature_dims,
                       embedding_size=embedding_size_rnn_output,
                       hidden_size=hidden_size_rnn_output//2,
                       output_size=edge_feature_dims,
                       out_middle_layer=out_edge_level)
    return rnn, output


class GRU_plain(torch.nn.Module):
    def __init__(self, input_size, out_middle_layer, embedding_size, hidden_size, output_size, node_lvl=False):
        super(GRU_plain, self).__init__()
        self.hidden_size = hidden_size
        self.node_lvl = node_lvl
        self.out_middle_layer = out_middle_layer
        self.num_layers = 2
        self.input = torch.nn.Embedding(
            input_size, embedding_size)  # embedding layer
        self.rnn = torch.nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=self.num_layers,
                                batch_first=True)

        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, out_middle_layer),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(out_middle_layer, output_size))

        if node_lvl:
            self.node_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, out_middle_layer),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(out_middle_layer, node_feature_dims))

        # initialize:
        self.hidden = None  # need initialize before forward run
        self.init_layers()


    def init_layers(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.25).to(device)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(
                    param, gain=torch.nn.init.calculate_gain('sigmoid')).to(device)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(
                    m.weight.data, gain=torch.nn.init.calculate_gain('leaky_relu')).to(device)

    def init_hidden(self, batch_size): return torch.zeros(
        (self.num_layers, batch_size, self.hidden_size), requires_grad=True).to(device)

    def init_hidden_rand(self, batch_size): return torch.rand(
        (self.num_layers, batch_size, self.hidden_size), requires_grad=True).to(device)

    def forward(self, input_raw, input_len=None):
        input = self.input(input_raw)  # embedding, no activation
        input = torch.nn.utils.rnn.pack_padded_sequence(
            input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        output_raw = torch.nn.utils.rnn.pad_packed_sequence(
            output_raw, batch_first=True)[0]
        # return hidden state at each time step
        output_raw_1 = self.output(output_raw)

        if self.node_lvl:
            node_pred = self.node_mlp(output_raw)
            return output_raw_1, node_pred

        return output_raw_1


def train_rnn_epoch(rnn, output, data_loader_, optimizer_rnn, optimizer_output, node_weights, edge_weights, device):

    rnn.train()
    output.train()
    loss_sum, loss_sum_edges, loss_sum_nodes = 0, 0, 0

    for batch_idx, data in enumerate(data_loader_):

        rnn.zero_grad()
        output.zero_grad()

        loss, edge_loss, node_loss = fit_batch(data, rnn, output)

        loss.backward(retain_graph=True)
        optimizer_output.step()
        optimizer_rnn.step()

        loss_sum += loss.data
        loss_sum_edges += edge_loss.data
        loss_sum_nodes += node_loss.data

    return loss_sum / (batch_idx + 1), loss_sum_edges / (batch_idx + 1), loss_sum_nodes / (batch_idx + 1)


def fit_batch(data, rnn, output):
    # ([bs, max_num_node, max_prev_node, edge_feature])
    x_unsorted = data['x'].float()
    y_unsorted = data['y'].float()

    # ([bs, max_num_node, node_feature])
    x_nodes_unsorted = data['x_node_attr'].float()
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
    # ([bs, max_seq_l, 8*4])
    x = torch.index_select(x_unsorted_for_nn, 0, sort_index)
    y = torch.index_select(y_unsorted_for_nn, 0, sort_index)
    # ([bs, max_seq_l, node_f])
    x_nodes = torch.index_select(x_nodes_unsorted, 0, sort_index)
    y_nodes = torch.index_select(
        y_nodes_unsorted, 0, sort_index)  # NODE TARGETS
    y_reshape = torch.nn.utils.rnn.pack_padded_sequence(
        y, y_len, batch_first=True).data
    idx = [i for i in range(y_reshape.size(0) - 1, -1, -1)]
    idx = torch.LongTensor(idx)
    # inverts the rows order of y_reshape
    y_reshape = y_reshape.index_select(0, idx)
    y_reshape = y_reshape.view(y_reshape.size(
        0), max_prev_node, edge_feature_dims)
    output_x = torch.cat((torch.ones(y_reshape.size(
        0), 1, edge_feature_dims), y_reshape[:, 0:-1, :]), dim=1)
    output_y = y_reshape
    output_y_len = []
    output_y_len_bin = np.bincount(np.array(y_len))
    for i in range(len(output_y_len_bin) - 1, 0, -1):  # countdown from len(output_y_len_bin)
        # count how many times y_len is above i
        count_temp = np.sum(output_y_len_bin[i:])
        if i == max_num_node:
            output_y_len.extend([min(max_prev_node, y.size(2))] * count_temp)
        else:
            output_y_len.extend([min(i, y.size(2))] * count_temp)

    x = torch.cat((x_nodes, x), 2)  # INPUT FOR NODE LVL
    x = torch.tensor(x).to(device)
    output_x = torch.tensor(output_x).to(device)

    y_nodes = torch.tensor(y_nodes).to(device)
    output_y = torch.tensor(output_y).to(device)

    # OUTPUTS
    h, node_prediction = rnn(x, input_len=y_len)

    h = torch.nn.utils.rnn.pack_padded_sequence(
        h, y_len, batch_first=True).data  # get packed hidden vector
    idx = [i for i in range(h.size(0) - 1, -1, -1)]
    idx = torch.tensor(torch.LongTensor(idx)).to(device)
    h = h.index_select(0, idx)

    hidden_null = torch.tensor(torch.zeros(
        rnn.num_layers - 1, h.size(0), h.size(1))).to(device)
    # num_layers, batch_size, hidden_size
    output.hidden = torch.cat(
        (h.view(1, h.size(0), h.size(1)), hidden_null), dim=0)

    y_pred = output(output_x, input_len=output_y_len)

    # OUTPUT LAYERS
    
    # pack and pad predicted edges
    y_pred = torch.nn.utils.rnn.pack_padded_sequence(
        y_pred, output_y_len, batch_first=True)
    y_pred = torch.nn.utils.rnn.pad_packed_sequence(
        y_pred, batch_first=True)[0]

    # pack and pad real edges
    output_y = torch.nn.utils.rnn.pack_padded_sequence(
        output_y, output_y_len, batch_first=True)
    output_y = torch.nn.utils.rnn.pad_packed_sequence(
        output_y, batch_first=True)[0]

    # pack and pad predicted nodes
    node_prediction = torch.nn.utils.rnn.pack_padded_sequence(
        node_prediction, y_len, batch_first=True)
    node_prediction = torch.nn.utils.rnn.pad_packed_sequence(
        node_prediction, batch_first=True)[0]

    # pack and pad real nodes
    y_nodes = torch.nn.utils.rnn.pack_padded_sequence(
        y_nodes, y_len, batch_first=True)
    y_nodes = torch.nn.utils.rnn.pad_packed_sequence(
        y_nodes, batch_first=True)[0]

    # permutations of predicted edg/nodes
    y_pred = y_pred.permute(0, 2, 1)
    node_prediction = node_prediction.permute(0, 2, 1)

    # permutations of real edg/nodes
    output_y = output_y.permute(0, 2, 1)
    y_nodes = y_nodes.permute(0, 2, 1)

    values, indices_edges = torch.max(output_y, 1)
    values, indices_nodes = torch.max(y_nodes, 1)

    # weight=edge_weights[1:].to(torch.float32).to(device)
    edge_loss = F.cross_entropy(y_pred, indices_edges, reduction='mean')
    # weight=node_weights.to(torch.float32).to(device))
    node_loss = F.cross_entropy(
        node_prediction, indices_nodes, reduction='mean')
    loss = edge_loss + node_loss
    return loss, edge_loss, node_loss
