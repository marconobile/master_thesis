import os
from rdkit import Chem
import torch
import torch.nn.functional as F
from mappings import *
from torch_geometric.data import Data
from torch_sparse import coalesce
from torch.utils.data import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_dense_adj
import numpy as np
import networkx as nx
from torch import nn
from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import lightning as L
import torch.nn.init as init
import torchmetrics
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")


def save_smiles(smiles, path, filename, ext='.txt'):
    '''
    saves smiles in a file at path
    extension can be provided in filename or as separate arg
    args:
        - smiles str iterable 
        - path directory where to save smiles 
        - filename name of the file, must not have extension
    '''
    path_to_file = os.path.join(path, filename)
    filename_ext = os.path.splitext(path_to_file)[-1].lower()
    if not filename_ext:
        if ext not in ['.txt', '.smiles']:
            raise f"extension {ext} not valid"
        path_to_file += ext

    # path_to_file = generate_file(path, filename)
    with open(path_to_file, "w+") as f:
        f.writelines("%s\n" % smi for smi in smiles)


def pyg2rdkit(dataset):
    def numpy_to_rdkit(adj, nf, ef, sanitize=False):
        """
        Converts a molecule from numpy to RDKit format.
        :param adj: binary numpy array of shape (N, N) 
        :param nf: numpy array of shape (N, F)
        :param ef: numpy array of shape (N, N, S)
        :param sanitize: whether to sanitize the molecule after conversion
        :return: an RDKit molecule
        """
        if Chem is None:
            raise ImportError('`numpy_to_rdkit` requires RDKit.')
        mol = Chem.RWMol()
        for nf_ in nf:
            # atomic_num = torch.argmax(nf_).item()
            atomic_num = int(nf_)
            mol.AddAtom(Chem.Atom(num2atom[atomic_num]))

        for i, j in zip(*np.triu_indices(adj.shape[-1])):
            if i != j and adj[i, j] == adj[j, i] == 1 and not mol.GetBondBetweenAtoms(int(i), int(j)):
                bond_type_1 = num2bond[int(ef[i, j, 0])]
                bond_type_2 = num2bond[int(ef[j, i, 0])]
                if bond_type_1 == bond_type_2:
                    mol.AddBond(int(i), int(j), bond_type_1)

        mol = mol.GetMol()
        if sanitize:
            Chem.SanitizeMol(mol)
        return mol

    mols_ = []
    for i, obs in enumerate(dataset):
        ef_temp = torch.squeeze(to_dense_adj(
            edge_index=obs.edge_index, batch=None, edge_attr=obs.edge_attr), 0)
        ef = torch.zeros((ef_temp.shape[0], ef_temp.shape[1], 1))
        adj = torch.zeros((ef_temp.shape[0], ef_temp.shape[1]))

        for row in range(ef_temp.shape[0]):
            for col in range(ef_temp.shape[1]):
                if int(torch.sum(ef_temp[row, col]).item()) != 0:
                    ef[row, col, 0] = torch.argmax(ef_temp[row, col]).item()
                    adj[row, col] = 1

        ef = np.array(ef)
        adj = np.array(adj)

        adj = adj.astype(int)
        ef = ef.astype(int)

        nf = np.array([torch.argmax(row).item() for row in obs.x])
        nf = np.expand_dims(nf, 1)
        nf = nf.astype(int)
        mols_.append(numpy_to_rdkit(adj, nf, ef))

    return mols_


# ORIGINAL NX FUNCTION
def to_undirected(graph):
    """Returns an undirected view of the graph `graph`.

    Identical to graph.to_undirected(as_view=True)
    Note that graph.to_undirected defaults to `as_view=False`
    while this function always provides a view.
    """
    return graph.to_undirected(as_view=True)


def encode_adj(adj, original, max_prev_node, edge_feature_dims):
    '''
    :param adj: A of current g with edge features as els : (V, V, 4)
    :param original: plain A of current g (without edge features: a binary matrix)
    :param max_prev_node: number of nodes of current graph - 1
    :return: encoded structure for the edges
    '''

    n = original.shape[0] - 1  # N - 1 of the current graph
    temp = np.zeros((n, max_prev_node, edge_feature_dims))

    original_tril = np.tril(original, k=-1)  # lower tri of original A
    original_tril_idx = np.nonzero(original_tril)

    # begin by setting all as absent
    for r in range(temp.shape[0]):
        for c in range(temp.shape[1]):
            temp[r, c, 0] = 1

    for index in range(len(original_tril_idx[0])):
        i = original_tril_idx[0][index]
        j = original_tril_idx[1][index]
        temp[i - 1, j, :] = np.concatenate((np.array([0.]), adj[i, j, :]), 0)
        # [i - 1, j ]  since we drop first row of A in the encoding, we need to 'move up' every row-idx

    adj_output = np.zeros((n, max_prev_node, edge_feature_dims))

    # flip
    for i in range(0, n):
        adj_output[i, :i + 1, :] = np.flip(temp[i, :i + 1, :], 0)

    return adj_output


class Graph_sequence_sampler_pytorch(torch.utils.data.Dataset):
    '''
    returns : dictionary containing input/output nodes, input/output edges
    '''

    def __init__(self, Graph_list, node_attr_list, adj_all, max_num_node, max_prev_node):
        '''
        Graph_list: list of undirected networkx graphs
        node_attr_list: list of node matrices
        adj_all: list of A(s) with edge features as elements a_ij [NxNxEf]
        max_num_node : max number of possible nodes in a graph
        max_prev_node : max previous node that looks back (to lock back at)
        '''

        # list of multidim np.arrays (As) already in edge_feature form [V, V , node_f]
        self.adj_all = adj_all
        self.len_all = []  # V for each G
        self.node_attr_list = node_attr_list
        self.graph_list = Graph_list  # list of undirected nx graphs
        self.len_all = [G.number_of_nodes() for G in Graph_list] # timesteps of node rnn for each G                    
        self.max_num_node = max_num_node
        self.max_prev_node = max_prev_node
        self.edge_feature_dims = 5
        self.node_feature_dims = 12
        self.dataset = [self.preprocess_obs_i(i) for i in range(self.__len__())]

    def __getitem__(self, idx): return self.dataset[idx]

    def __len__(self): return len(self.adj_all)

    def preprocess_obs_i(self, idx):
        # edge encoding:
        adj_copy = np.asarray(self.adj_all[idx]).copy()
        adj_copy = np.squeeze(adj_copy)  # adj_copy had bs as first dim
        x_batch = np.zeros((self.max_num_node, self.max_prev_node, self.edge_feature_dims))
        y_batch = np.zeros((self.max_num_node, self.max_prev_node, self.edge_feature_dims))

        # A without edge features of the current g
        original_a = np.asarray(nx.adjacency_matrix(self.graph_list[idx]).todense())
        # original_a = np.asarray(nx.from_numpy_array(self.graph_list[idx]))  # A without edge features of the current g
        adj_encoded = encode_adj(adj=adj_copy, original=original_a, max_prev_node=self.max_prev_node, edge_feature_dims=self.edge_feature_dims)

        x_batch[0, :, :] = 1
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded

        for r in range(y_batch.shape[0]):
            for c in range(y_batch.shape[1]):
                if np.sum(y_batch[r, c, :]) == 0:
                    y_batch[r, c, 0] = 1

        # node encoding:
        node_attr_list_copy = np.asarray(self.node_attr_list[idx]).copy()
        x_node_attr = np.zeros((self.max_num_node, self.node_feature_dims))
        y_node_attr = np.zeros((self.max_num_node, self.node_feature_dims))

        # input nodes:
        x_node_attr[0, :] = 1
        x_node_attr[1:node_attr_list_copy.shape[0], :] = node_attr_list_copy[:-1]
        # output nodes:
        y_node_attr[:node_attr_list_copy.shape[0], :] = node_attr_list_copy

        # len_batch = number of nodes of current g
        return {'x': torch.from_numpy(x_batch).to(device), 
                'y': torch.from_numpy(y_batch).to(device), 
                'len': torch.tensor(node_attr_list_copy.shape[0], dtype=torch.int64).to(device), 
                'x_node_attr': torch.from_numpy(x_node_attr).to(device), 
                'y_node_attr': torch.from_numpy(y_node_attr).to(device)
                }


def process_subset(subset, max_num_node, max_prev_node):
    '''
    :param subset: dataset for training or dataset for testing
    :param max_num_node: max num of nodes in the set of graphs
    :param max_prev_node: max_num_node - 1
    :return: Graph_sequence_sampler_pytorch data object
    '''

    G_list = []         # list of undirected nx graphs
    node_attr_list = []  # list of node matrices
    adj_all = []     # list of A(s) with edge features as elements a_ij [NxNxEf]
    for g in subset:
        node_attr_list.append(g.x)
        G_list.append(to_undirected(to_networkx(g)))
        adj_all.append(to_dense_adj(edge_index=g.edge_index,
                       batch=None, edge_attr=g.edge_attr))

    return Graph_sequence_sampler_pytorch(Graph_list=G_list, node_attr_list=node_attr_list, adj_all=adj_all,
                                          max_num_node=max_num_node, max_prev_node=max_prev_node)


def create_train_val_dataloaders(trainset, valset, bs, num_workers=1):
    '''
    for supervised training takes as input:
    - dataset: a list of pyg Data obs,
    - max number of nodes of the loaded graphs,
    - max_prev_node = (max number of nodes-1)
    '''
    train_set = process_subset(trainset, max_num_node, max_prev_node)
    val_set = process_subset(valset, max_num_node, max_prev_node)
    train_dataset_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=num_workers)# , pin_memory=True)
    val_dataset_loader = DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=num_workers)# , pin_memory=True)
    return train_dataset_loader, val_dataset_loader


def mols_from_file(pathfile: str, drop_none: bool = False):
    '''
    takes as input a path/to/file.ext 
    where ext can be:
    .sdf, .csv, .txt, .smiles
    it returns all mols from file
    if drop_none: drops all mols non valid for rdkit
    '''
    filename_ext = os.path.splitext(pathfile)[-1].lower()
    if filename_ext in ['.sdf']:
        suppl = Chem.SDMolSupplier(pathfile)
    elif filename_ext in ['.csv', '.txt', '.smiles']:
        suppl = Chem.SmilesMolSupplier(pathfile, titleLine=False)
    else:
        raise TypeError(f"{filename_ext} not supported")
    if drop_none:
        return [x for x in suppl if x is not None]
    return [x for x in suppl]


def rdkit2pyg(mols):
    '''
    #! TODO: multiprocess
    :param mols: iterable of rdkit mols
    :return: list of PyG data objs with one-hot node/edge features
    '''
    data_list = []
    for mol in mols:
        if mol is None:
            continue
        N = mol.GetNumAtoms()
        type_idx = []
        for atom in mol.GetAtoms():
            type_idx.append(atom2num[atom.GetSymbol()])
        x = F.one_hot(torch.tensor(type_idx), num_classes=len(atom2num))
        row, col, bond_idx = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            bond_idx += 2 * [bond2num[bond.GetBondType()]]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = F.one_hot(torch.tensor(bond_idx).to(
            torch.int64), num_classes=len(bond2num)).to(torch.float)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    return data_list


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

        self.ad_hoc_init()

    def ad_hoc_init(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name: nn.init.constant_(param, 0.0)
            elif 'weight' in name: nn.init.kaiming_normal_(param, nonlinearity='sigmoid')

        if self.node_lvl:
            torch.nn.init.kaiming_normal_(self.node_mlp1.weight, nonlinearity='leaky_relu')
            torch.nn.init.kaiming_normal_(self.node_mlp2.weight, nonlinearity='leaky_relu')
            self.node_mlp2.weight.data *= 0.01
            torch.nn.init.zeros_(self.node_mlp2.bias)
        else:
            torch.nn.init.kaiming_normal_(self.output1.weight, nonlinearity='leaky_relu')
            torch.nn.init.kaiming_normal_(self.output2.weight, nonlinearity='leaky_relu')
            self.output2.weight.data *= 0.01
            torch.nn.init.zeros_(self.output2.bias)

    def init_hidden(self, batch_size): return torch.zeros((self.num_layers, batch_size, self.hidden_size), requires_grad=True,device=device)

    def init_hidden_rand(self, batch_size): return torch.rand((self.num_layers, batch_size, self.hidden_size), requires_grad=True,device=device)

    def forward(self, input_raw, pack=False, input_len=None):

        input_emb = self.embedding(input_raw)
        input_emb = self.embeddingLRelu(input_emb)

        if pack:
            input = nn.utils.rnn.pack_padded_sequence(input_emb, input_len, batch_first=True)
            output_raw, self.hidden = self.rnn(input, self.hidden)
            output_raw = nn.utils.rnn.pad_packed_sequence(output_raw, batch_first=True)[0]
        else:
            output_raw, self.hidden = self.rnn(input_emb, self.hidden)

        if self.node_lvl: output_raw = 1/2*input_emb + 1/2*output_raw

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


class GraphRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        num_layers = 4

        embedding_size_rnn = 256 * 4
        hidden_size_rnn = 256 * 4

        embedding_size_rnn_output = 256 * 4
        hidden_size_rnn_output = 256 * 4
        out_edge_level = hidden_size_rnn_output

        self.rnn = GRU_plain(input_size=node_feature_dims + edge_feature_dims * max_prev_node,
                             num_layers=num_layers,
                             embedding_size=embedding_size_rnn,
                             hidden_size=hidden_size_rnn,
                             output_size=hidden_size_rnn_output,
                             node_lvl=True,
                             out_middle_layer=hidden_size_rnn_output)

        self.output = GRU_plain(input_size=edge_feature_dims,
                                num_layers=num_layers,
                                embedding_size=embedding_size_rnn_output,
                                hidden_size=hidden_size_rnn_output,
                                output_size=edge_feature_dims,
                                out_middle_layer=out_edge_level)

        self.node_weights = torch.tensor(nweights_list, device=device, dtype=torch.float32)
        self.edge_weights = torch.tensor(bweights_list, device=device, dtype=torch.float32)
        self.ce_nodes = torch.nn.CrossEntropyLoss(self.node_weights)
        self.ce_edges = torch.nn.CrossEntropyLoss(self.edge_weights)

    def forward(self, data, train_edge_lvl = False):        

        with torch.no_grad():
            # ([bs, max_num_node, max_prev_node, edge_feature])
            y_unsorted = data['y'].float()
            x_unsorted = data['x'].float()

            # ([bs, max_num_node, node_feature])
            y_nodes_unsorted = data['y_node_attr'].float()
            x_nodes_unsorted = data['x_node_attr'].float()

            y_len_unsorted = data['len']  # list of seq_len of each g, len()=bs
            # pick max_seq_length of the current batch
            y_len_max = int(torch.max(data['len']).item())
            x_unsorted = x_unsorted[:, :y_len_max, :, :]
            y_unsorted = y_unsorted[:, :y_len_max, :, :]

            x_nodes_unsorted = x_nodes_unsorted[:, :y_len_max, :]
            y_nodes_unsorted = y_nodes_unsorted[:, :y_len_max, :]

            x_unsorted_for_nn = x_unsorted.flatten(-2,-1) # torch.reshape(x_unsorted, (x_unsorted.shape[0], x_unsorted.shape[1], x_unsorted.shape[2] * x_unsorted.shape[3]))
            y_unsorted_for_nn = y_unsorted.flatten(-2,-1) # torch.reshape(y_unsorted, (x_unsorted.shape[0], x_unsorted.shape[1], x_unsorted.shape[2] * x_unsorted.shape[3]))

            y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
            y_len = y_len.cpu()  # ([bs, max_seq_l, 8*4])
            x = torch.index_select(x_unsorted_for_nn, 0, sort_index)
            # ([bs, max_seq_l, node_f])
            y = torch.index_select(y_unsorted_for_nn, 0, sort_index)
            x_nodes = torch.index_select(x_nodes_unsorted, 0, sort_index)
            y_nodes = torch.index_select(y_nodes_unsorted, 0, sort_index)  
            
            # NODE TARGETS
            y_reshape, y_seq_len, _, _  = pack_padded_sequence(y, y_len, batch_first=True)
            idx = torch.tensor([i for i in range(y_reshape.size(0) - 1, -1, -1)], device=device, dtype=torch.long)
            # inverts the rows order of y_reshape
            y_reshape = y_reshape.index_select(0, idx)
            y_reshape = y_reshape.view(y_reshape.size(0), max_prev_node, edge_feature_dims)
            output_x = torch.cat((torch.ones(y_reshape.size(0), 1, edge_feature_dims, device=device), y_reshape[:, 0:-1, :]), dim=1)
            output_y = y_reshape
            output_y_len = []
            output_y_len_bin = torch.bincount(y_len)
            # countdown from len(output_y_len_bin)
            for i in range(len(output_y_len_bin) - 1, 0, -1):
                # count how many times y_len is above i
                count_temp = torch.sum(output_y_len_bin[i:])
                if i == max_num_node:
                    output_y_len.extend([min(max_prev_node, y.size(2))] * count_temp)
                else: output_y_len.extend([min(i, y.size(2))] * count_temp)
            x = torch.cat((x_nodes, x), 2)  # INPUT FOR NODE LVL

        # OUTPUTS
        self.rnn.hidden = self.rnn.init_hidden(batch_size=x_unsorted_for_nn.size(0))
        h, node_prediction = self.rnn(x, pack=True, input_len=y_len)

        node_prediction = pack_padded_sequence(node_prediction, y_len, batch_first=True)
        y_nodes = pack_padded_sequence(y_nodes, y_len, batch_first=True)
        node_loss = self.ce_nodes(node_prediction[0], y_nodes[0])
        total_loss = node_loss

        edge_loss = torch.tensor([torch.nan])
        if train_edge_lvl:            
            h = pack_padded_sequence(h, y_len, batch_first=True).data  # get packed hidden vector
            idx = torch.tensor([i for i in range(h.size(0) - 1, -1, -1)], device=device, dtype=torch.long)
            h = h.index_select(0, idx)

            # num_layers, batch_size, hidden_size
            hidden_null = torch.zeros(self.rnn.num_layers - 1, h.size(0), h.size(1), device=device)
            self.output.hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null), dim=0)
            y_pred = self.output(output_x, pack=True, input_len=output_y_len)

            # OUTPUT LAYERS
            y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
            output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
            edge_loss = self.ce_edges(y_pred[0], output_y[0])
            total_loss +=  edge_loss*.1

        return {"total_loss": total_loss, "edge_loss": edge_loss, "node_loss": node_loss}
    

class LightModule(L.LightningModule):
    def __init__(self, lr, steps_per_epoch, epochs):        
        super().__init__()
        self.model = GraphRNNModel()
        self.lr = lr
        self.steps_per_epoch, self.epochs = steps_per_epoch, epochs
        self.train_edge_lvl = False
        # self.train_node_acc = torchmetrics.Accuracy( task="multiclass", num_classes=12)
        # self.val_node_acc = torchmetrics.Accuracy( task="multiclass", num_classes=5)
        # self.train_edge_acc = torchmetrics.Accuracy(task="multiclass", num_classes=12)
        # self.val_edge_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)

    def forward(self, x):         
        return self.model(x, self.train_edge_lvl)

    def _shared_step(self, batch): 
        return self.forward(batch)

    def training_step(self, batch, batch_idx):
        return_dict = self._shared_step(batch)    
        self.log("total_loss", return_dict["total_loss"].item(), prog_bar=True)
        self.log("edge_loss", return_dict["edge_loss"].item(), prog_bar=True)
        node_loss = return_dict["node_loss"].item()
        if node_loss < 1e-2:
            self.train_edge_lvl = True
        self.log("node_loss", node_loss, prog_bar=True)        
        return return_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        pass
        # loss, y, predictions = self._shared_step(batch)
        # self.val_acc(predictions, y)
        # self.log("validation_loss", loss, prog_bar=True)
        # self.log("val_acc", self.val_acc, prog_bar=True) # not at every batch but at each epoch

    def test_step(self, batch, batch_idx):
        pass
        # loss, y, predictions = self._shared_step(batch)
        # self.test_acc(predictions, y)
        # self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.lr)
        # scheduler = OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs)
        # scheduler =  CosineAnnealingLR(optimizer, self.steps_per_epoch * self.epochs)
        return [optimizer] #, [scheduler]

    @torch.no_grad()
    def _generate_single_obs(self, test_batch_size=1):        
        self.model.rnn.hidden = self.model.rnn.init_hidden(test_batch_size) #! rand    
        x_step = torch.ones((test_batch_size, 1, max_prev_node * edge_feature_dims + node_feature_dims), device=device)
        x_list, edg_attr_list, edg_idx_list = [], [], [] # initialize empty lists for Data() object
        
        for i in range(max_num_node): # Node RNN for-loop
            h, node_prediction = self.model.rnn(x_step)
            node_prediction_argmax = F.one_hot(node_prediction.argmax(), num_classes=node_feature_dims)
            x_list.append(node_prediction_argmax)

            # reset and update input for next iteration
            x_step = torch.zeros((test_batch_size, 1, max_prev_node * edge_feature_dims + node_feature_dims), device=device)
            x_step[:, :, :node_feature_dims] = node_prediction_argmax.data

            # init Edge/Abs lvl
            hidden_null = torch.zeros((self.model.rnn.num_layers - 1, h.size(0), h.size(2))).to(device)
            self.model.output.hidden = torch.cat((h.permute(1, 0, 2), hidden_null), dim=0).to(device)

            # token Edge lvl - randn best result
            output_x_step = torch.ones(test_batch_size, 1, edge_feature_dims).to(device)
            edge_rnn_step = 0
            idx = [k for k in range(i, -1, -1)] # this list is used to create edg_idx
            for j in range(min(max_prev_node, i + 1)): # Edge/Abs RNN for-loop          
                output_y_pred_step_out = self.model.output(output_x_step) # prediction for each and every prev node
                # self.model.output.hidden = self.model.output.hidden.data.to(device)
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
    def generate_mols(self, N):
        self.model.rnn.to(device)
        self.model.output.to(device)
        self.model.rnn.eval()
        self.model.output.eval()
        to_draw = []
        for idx in range(N):
            print(f"{idx+1}/{N}", end='\r')
            obs = self._generate_single_obs(test_batch_size=1)
            to_draw.append(obs)

        smiles_ = pyg2rdkit(to_draw)
        filename = f"generating_{N}.smiles"
        smiles_ = [Chem.MolToSmiles(m) for m in smiles_]
        save_smiles(smiles_, ".", filename, "smiles")
