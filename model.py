import os
import torch
from mappings import node_feature_dims, edge_feature_dims, max_prev_node
import torch.nn as nn
import torch.nn.functional as F


cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")

def get_generator():

    num_layers = 4

    embedding_size_rnn = 256 *4
    hidden_size_rnn = 256 *4

    embedding_size_rnn_output = 256 *3
    hidden_size_rnn_output = 256 * 3
    out_edge_level = hidden_size_rnn_output

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

        # Embedding
        self.embedding = nn.Linear(input_size, embedding_size)
        self.embeddingLRelu = nn.LeakyReLU()

        # RNN
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True)

        # output: 
        # if node lvl: returns processed h_to_pass, dims: edgelvl hs
        # if edge lvl: returns edge unnormalized logits, dims: ef
        self.output1 = nn.Linear(hidden_size, out_middle_layer, bias=False)
        self.outputNorm = nn.LayerNorm(out_middle_layer)
        self.outputLRelu = nn.LeakyReLU()
        self.output2 = nn.Linear(out_middle_layer, output_size)
        
        # node unnormalized logits
        if node_lvl:
            self.node_mlp1 = nn.Linear(hidden_size, out_middle_layer, bias=False)
            self.nodeNorm = nn.LayerNorm(out_middle_layer)
            self.nodeLRelu = nn.LeakyReLU()
            self.node_mlp2 = nn.Linear(out_middle_layer, node_feature_dims)          

    def ad_hoc_init(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name: nn.init.constant_(param, 0.0)
            elif 'weight' in name: nn.init.kaiming_normal_(param, nonlinearity='sigmoid')

        if self.node_lvl:
            torch.nn.init.kaiming_normal_(self.node_mlp1.weight, nonlinearity='leaky_relu')
            torch.nn.init.kaiming_normal_(self.node_mlp2.weight, nonlinearity='leaky_relu')            
            self.node_mlp2.weight.data *= 0.01  
            torch.nn.init.zeros_(self.node_mlp2.bias)         
            # self.node_mlp2.bias = torch.nn.Parameter(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.], dev         
        else:
            torch.nn.init.kaiming_normal_(self.output1.weight, nonlinearity='leaky_relu')
            torch.nn.init.kaiming_normal_(self.output2.weight, nonlinearity='leaky_relu')
            self.output2.weight.data *= 0.01
            torch.nn.init.zeros_(self.output2.bias)     
            # self.output2.bias = torch.nn.Parameter(torch.tensor([1., 0., 0., 0., 0.], device=device))             

    def init_hidden(self, batch_size): return torch.zeros((self.num_layers, batch_size, self.hidden_size), requires_grad=True).to(device)
    def init_hidden_rand(self, batch_size): return torch.rand((self.num_layers, batch_size, self.hidden_size), requires_grad=True).to(device)

    def get_activation_layers(self):
        return {"embeddingLRelu": self.embeddingLRelu, "outputLRelu": self.outputLRelu, "nodeLRelu": self.nodeLRelu}

    def forward(self, input_raw, pack=False, input_len=None):

        input_emb = self.embedding(input_raw)
        input_emb = self.embeddingLRelu(input_emb)

        if pack: 
            input = nn.utils.rnn.pack_padded_sequence(input_emb, input_len, batch_first=True)
            output_raw, self.hidden = self.rnn(input, self.hidden)        
            output_raw = nn.utils.rnn.pad_packed_sequence(output_raw, batch_first=True)[0]
        else:
            output_raw, self.hidden = self.rnn(input_emb, self.hidden)        
        
        if self.node_lvl:
            output_raw = input_emb + output_raw

        output_raw_1 = self.output1(output_raw)
        output_raw_1 = self.outputNorm(output_raw_1)
        output_raw_1 = self.outputLRelu(output_raw_1)
        output_raw_1 = self.output2(output_raw_1)

        if self.node_lvl:
            node_pred = self.node_mlp1(output_raw)
            node_pred = self.nodeNorm(node_pred) 
            node_pred = self.nodeLRelu(node_pred)
            node_pred = self.node_mlp2(node_pred)
            return output_raw_1, node_pred

        return output_raw_1

    def get_save_path(self): return os.getcwd() + "./weights/"
    def save(self, epoch):
        self.save_path = os.getcwd() + "./weights/"
        checkpoint = {'model_state_dict': self.state_dict(),}
        if self.node_lvl: torch.save(checkpoint, self.save_path + f'nodeRNN_checkpoint_{epoch}.pth')
        else: torch.save(checkpoint, self.save_path + f'edgeRNN_checkpoint_{epoch}.pth')

