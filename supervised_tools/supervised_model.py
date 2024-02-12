# from args import Args
import torch


node_feature_dims = 12
edge_feature_dims = 5
max_num_node = 88
max_prev_node = max_num_node - 1

def get_generator():
    # args = Args()
    num_layers = 2

    # NODE LEVEL AND ABSENCE NET embeddings and hidden sizes
    embedding_size_rnn = 256
    hidden_size_rnn = 256

    # EDGE LEVEL embeddings and hidden sizes
    embedding_size_rnn_output = 128
    hidden_size_rnn_output = 128
    out_edge_level = hidden_size_rnn_output

    rnn = GRU_plain(input_size=node_feature_dims + edge_feature_dims * max_prev_node,                                    
                    embedding_size=embedding_size_rnn,
                    hidden_size=hidden_size_rnn, num_layers=num_layers, has_input=True,
                    has_output=True, output_size=hidden_size_rnn_output, node_lvl=True,
                    out_middle_layer=hidden_size_rnn_output)

    absence_net = GRU_plain(input_size=edge_feature_dims, embedding_size=embedding_size_rnn_output,
                            hidden_size=hidden_size_rnn_output, num_layers=num_layers, has_input=True,
                            has_output=True, output_size=1, node_lvl=False, out_middle_layer=out_edge_level)

    output = GRU_plain(input_size=edge_feature_dims, embedding_size=embedding_size_rnn_output,
                       hidden_size=hidden_size_rnn_output, num_layers=num_layers, has_input=True,
                       has_output=True, output_size=edge_feature_dims - 1, node_lvl=False,
                       out_middle_layer=out_edge_level)

    return rnn, output, absence_net


class GRU_plain(torch.nn.Module):
    def __init__(self, input_size, out_middle_layer, embedding_size, hidden_size, num_layers, has_input=True,
                 has_output=False, output_size=None, node_lvl=False):
        super(GRU_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output
        self.node_lvl = node_lvl
        self.out_middle_layer = out_middle_layer
        self.cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        # self.args = Args()

        if has_input:
            self.input = torch.nn.Linear(input_size, embedding_size)  # embedding layer
            self.rnn = torch.nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                                    batch_first=True)
        else:
            self.rnn = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                    batch_first=True)
        if has_output:
            self.output = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, out_middle_layer),
                torch.nn.ReLU(),
                torch.nn.Linear(out_middle_layer, output_size))

        if node_lvl:
            self.node_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, out_middle_layer),
                torch.nn.ReLU(),
                torch.nn.Linear(out_middle_layer, node_feature_dims))

        self.relu = torch.nn.ReLU()

        # initialize:
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.25).to(self.device)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param, gain=torch.nn.init.calculate_gain('sigmoid')).to(self.device)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data,
                                                              gain=torch.nn.init.calculate_gain(
                                                                  'relu')).to(self.device)

    def init_hidden(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def init_hidden_rand(self, batch_size):
        return torch.rand((self.num_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)  # embedding
            input = self.relu(input)
        else:
            input = input_raw
        if pack:  # we take the embedded input
            input = torch.nn.utils.rnn.pack_padded_sequence(input, input_len, batch_first=True)

        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = torch.nn.utils.rnn.pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw_1 = self.output(output_raw)
        # return hidden state at each time step
        if self.node_lvl:
            node_pred = self.node_mlp(output_raw)
            return output_raw_1, node_pred
        else:
            return output_raw_1
