import os
import torch
from mappings import node_feature_dims, edge_feature_dims, max_prev_node
import torch.nn as nn
import torch.nn.functional as F


cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")

def get_generator():
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

class GRU_plain(nn.Module):
    def __init__(self, input_size, num_layers, out_middle_layer, embedding_size, hidden_size, output_size, node_lvl=False):
        super(GRU_plain, self).__init__()
        self.hidden_size, self.node_lvl, self.num_layers = hidden_size, node_lvl, num_layers          
        self.hidden = None  # need initialize before forward run

        # Layers
        self.embedding = nn.Linear(input_size, embedding_size)
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True)

        self.output1 = nn.Linear(hidden_size, out_middle_layer)
        self.output_norm = nn.LayerNorm(out_middle_layer)
        self.output2 = nn.Linear(out_middle_layer, output_size)
                    
        if node_lvl:
            self.node_mlp1 = nn.Linear(hidden_size, out_middle_layer)
            self.node_mlp2 = nn.Linear(out_middle_layer, node_feature_dims)          

    def ad_hoc_init(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name: nn.init.constant_(param, 0.25)
            elif 'weight' in name: nn.init.orthogonal_(param, gain=nn.init.calculate_gain('sigmoid'))

        # torch.nn.init.xavier_normal_(self.embedding.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
        if self.node_lvl:
            torch.nn.init.xavier_normal_(self.node_mlp1.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.node_mlp2.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))            
            self.node_mlp2.weight.data = self.node_mlp2.weight.data * .01  
            torch.nn.init.zeros_(self.node_mlp2.bias)                  
        else:
            torch.nn.init.xavier_normal_(self.output1.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
            torch.nn.init.xavier_normal_(self.output2.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
            self.output2.weight.data = self.output2.weight.data * .01
            torch.nn.init.zeros_(self.output2.bias)                  


    # def get_layers(self):
    #     if self.node_lvl: return self.embedding, self.rnn, self.output1, self.output2, self.node_mlp1, self.node_mlp2
    #     return self.embedding, self.rnn, self.output1, self.output2            

    def init_hidden(self, batch_size): return torch.zeros((self.num_layers, batch_size, self.hidden_size), requires_grad=True).to(device)
    def init_hidden_rand(self, batch_size): return torch.rand((self.num_layers, batch_size, self.hidden_size), requires_grad=True).to(device)

    def forward(self, input_raw, pack=False, input_len=None):
        # embed binary vector
        input = self.embedding(input_raw)
        # pack sequences
        if pack: 
            input = nn.utils.rnn.pack_padded_sequence(input, input_len, batch_first=True)
        # pass them thru the rnn
        output_raw, self.hidden = self.rnn(input, self.hidden)        

        output_raw = self.shared_norm(output_raw)
        # unpack sequences
        if pack: 
            output_raw = nn.utils.rnn.pad_packed_sequence(output_raw, batch_first=True)[0]
        # else: 
        output_raw_1 = self.output1(output_raw)
        output_raw_1 = F.leaky_relu(output_raw_1)
        output_raw_1 = self.output2(output_raw_1)

        if self.node_lvl:
            node_pred = self.node_mlp1(output_raw)
            node_pred = self.node_norm(node_pred)
            node_pred = F.leaky_relu(node_pred)
            node_pred = self.node_mlp2(node_pred)
            return output_raw_1, node_pred
        else: 
            return output_raw_1

    def get_save_path(self): return os.getcwd() + "./weights/"
    def save(self, epoch):
        self.save_path = os.getcwd() + "./weights/"
        checkpoint = {'model_state_dict': self.state_dict(),}
        if self.node_lvl: torch.save(checkpoint, self.save_path + f'nodeRNN_checkpoint_{epoch}.pth')
        else: torch.save(checkpoint, self.save_path + f'edgeRNN_checkpoint_{epoch}.pth')

