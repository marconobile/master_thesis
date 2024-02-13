import os
import torch
from mappings import node_feature_dims, edge_feature_dims, max_prev_node

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")

def get_generator():
    # NODE LEVEL AND ABSENCE NET embeddings and hidden sizes
    embedding_size_rnn = 256 *4
    hidden_size_rnn = 256 *4

    # EDGE LEVEL embeddings and hidden sizes
    embedding_size_rnn_output = 256 *3
    hidden_size_rnn_output = 256 * 3
    out_edge_level = hidden_size_rnn_output
    num_layers = 4

    rnn = GRU_plain(input_size=node_feature_dims + edge_feature_dims * max_prev_node,                    
        num_layers=num_layers,
        embedding_size=embedding_size_rnn,
        hidden_size=hidden_size_rnn,
        output_size=hidden_size_rnn_output,
        node_lvl=True,
        out_middle_layer=hidden_size_rnn_output)

    output = GRU_plain(input_size=edge_feature_dims,                    
        num_layers=num_layers,
        embedding_size=embedding_size_rnn_output,
        hidden_size=hidden_size_rnn_output,
        output_size=edge_feature_dims,
        out_middle_layer=out_edge_level)
    
    if cuda:
        rnn.to(device)
        output.to(device)
    return rnn, output

class GRU_plain(torch.nn.Module):
    def __init__(self, input_size, num_layers, out_middle_layer, embedding_size, hidden_size, output_size, node_lvl=False):
        super(GRU_plain, self).__init__()
        self.hidden_size, self.node_lvl, self.num_layers = hidden_size, node_lvl, num_layers          
        self.hidden = None  # need initialize before forward run

        # Layers
        self.rnn = torch.nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True)
        self.input = torch.nn.Linear(input_size, embedding_size)  # embedding layer
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, out_middle_layer),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(out_middle_layer, output_size))

        if node_lvl:
            self.node_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, out_middle_layer),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(out_middle_layer, node_feature_dims))
            
    def get_layers(self):
        if self.node_lvl:
            return self.rnn, self.input, self.output, self.node_mlp

    def init_hidden(self, batch_size): return torch.zeros((self.num_layers, batch_size, self.hidden_size), requires_grad=True).to(device)
    def init_hidden_rand(self, batch_size): return torch.rand((self.num_layers, batch_size, self.hidden_size), requires_grad=True).to(device)

    def forward(self, input_raw, pack=False, input_len=None):
        input = self.input(input_raw)  # embedding
        if pack: input = torch.nn.utils.rnn.pack_padded_sequence(input, input_len, batch_first=True)

        output_raw, self.hidden = self.rnn(input, self.hidden)        
        if pack: output_raw = torch.nn.utils.rnn.pad_packed_sequence(output_raw, batch_first=True)[0]
        else: output_raw_1 = self.output(output_raw) # return hidden state at each time step

        if self.node_lvl:
            node_pred = self.node_mlp(output_raw)
            return output_raw_1, node_pred
        else: return output_raw_1


    def save(self, epoch):
        path = os.getcwd() + "./weights/"
        checkpoint = {'model_state_dict': self.state_dict(),}
        if self.node_lvl:
            torch.save(checkpoint, path + f'nodeRNN_checkpoint_{epoch}.pth')
        else:
            torch.save(checkpoint, path + f'edgeRNN_checkpoint_{epoch}.pth')

