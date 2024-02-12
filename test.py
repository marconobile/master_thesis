import torch.nn.init as init
import torch.nn as nn
import networkx as nx
from supervised_tools.supervised_training import supervised_training
from supervised_tools.create_train_val_data import create_train_val_dataloaders
from torch_geometric.utils import to_dense_adj
import numpy as np
import os
from torch_geometric.data import Data
from torch_sparse import coalesce
import torch.nn.functional as F
from rdkit.Chem.rdchem import BondType as BT
from rdkit import Chem
from utils.setup import setup
import torch
print(torch.__version__)


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


def get_atoms_info(mols):
    atoms = set()
    max_num = 0
    for m in mols:
        if m.GetNumAtoms() > max_num:
            max_num = m.GetNumAtoms()
        for atom in m.GetAtoms():
            atoms.add(atom.GetSymbol())

    atom2num = {}

    for i, atomType in enumerate(atoms):
        atom2num[str(atomType)] = i

    num2atom = {v: k for k, v in atom2num.items()}
    return atom2num, num2atom, max_num


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
        edge_attr = F.one_hot(torch.tensor(bond_idx).to(torch.int64),
                              num_classes=len(bond2num)).to(torch.float)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)

        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    return data_list


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


class GRU_plain(torch.nn.Module):
    def __init__(self, input_size, num_layers, out_middle_layer, embedding_size, hidden_size, output_size, node_lvl=False):
        super(GRU_plain, self).__init__()
        self.hidden_size = hidden_size
        self.node_lvl = node_lvl
        self.out_middle_layer = out_middle_layer
        self.num_layers = num_layers
        self.input = torch.nn.Linear(
            input_size, embedding_size)  # embedding layer

        self.rnn = torch.nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=self.num_layers,
                                batch_first=True)

        # self.rnn = torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=device)

        # torch.nn.Sequential(
        #     torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=device),
        #     torch.nn.LayerNorm(hidden_size), 
        #     torch.nn.RNNCell(hidden_size, hidden_size, bias=True, nonlinearity='tanh', device=device),
        #     torch.nn.LayerNorm(hidden_size), 
        #     torch.nn.RNNCell(hidden_size, hidden_size, bias=True, nonlinearity='tanh', device=device),
        #     torch.nn.LayerNorm(hidden_size)
        # )

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
    #     self.init_layers()

    # def init_layers(self):
    #     for name, param in self.rnn.named_parameters():
    #         if 'bias' in name:
    #             torch.nn.init.constant_(param, 0.25).to(device)
    #         elif 'weight' in name:
    #             torch.nn.init.xavier_uniform_(
    #                 param, gain=torch.nn.init.calculate_gain('sigmoid')).to(device)

    #     for m in self.modules():
    #         if isinstance(m, torch.nn.Linear):
    #             # torch.nn.init.xavier_uniform_(
    #             #     m.weight.data, gain=torch.nn.init.calculate_gain('leaky_relu')).to(device)
    #             m.weight.data = m.weight.data *.01

    def init_hidden(self, batch_size): return torch.zeros(
        (self.num_layers, batch_size, self.hidden_size), requires_grad=True).to(device)

    def init_hidden_rand(self, batch_size): return torch.rand(
        (self.num_layers, batch_size, self.hidden_size), requires_grad=True).to(device)

    def forward(self, input_raw, pack=False, input_len=None):
        input_raw.to(device)
        input = self.input(input_raw)  # embedding

        if pack:  # we take the embedded input
            input = torch.nn.utils.rnn.pack_padded_sequence(
                input, input_len, batch_first=True)

        output_raw, self.hidden = self.rnn(input, self.hidden)        
        if pack:
            output_raw = torch.nn.utils.rnn.pad_packed_sequence(
                output_raw, batch_first=True)[0]
        else:
            # return hidden state at each time step
            output_raw_1 = self.output(output_raw)

        if self.node_lvl:
            node_pred = self.node_mlp(output_raw)
            return output_raw_1, node_pred
        else:
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

    y_pred = F.log_softmax(y_pred, dim=2)
    node_prediction = F.log_softmax(node_prediction, dim=2)

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

    # # weight=edge_weights[1:].to(torch.float32).to(device)
    # edge_loss = F.cross_entropy(y_pred, indices_edges, reduction='mean')
    # # weight=node_weights.to(torch.float32).to(device))
    # node_loss = F.cross_entropy(
    #     node_prediction, indices_nodes, reduction='mean')

    edge_loss = F.nll_loss(y_pred, indices_edges, reduction='mean')
    node_loss = F.nll_loss(node_prediction, indices_nodes, reduction='mean')

    loss = edge_loss + node_loss
    return loss, edge_loss, node_loss


def generate_single_obs(rnn, output, device, max_num_node, max_prev_node, test_batch_size=1):
    # initialize hidden state
    rnn.hidden = rnn.init_hidden(test_batch_size).to(
        device)  # init_hidden_rand
    # create node level token
    x_step = torch.ones((test_batch_size, 1, max_prev_node * edge_feature_dims + node_feature_dims),
                        requires_grad=False).to(device)
    # initialize empty lists for Data() object
    x_list = []
    edg_attr_list = []
    edg_idx_list = []

    # Node RNN for-loop
    for i in range(max_num_node):
        h, node_prediction = rnn(x_step)

        # arg-max + discretization
        idx_node_arg_max = torch.argmax(node_prediction, dim=-1).item()
        node_prediction_argmax = torch.zeros_like(
            node_prediction).to(device)  # torch.Size([1, 1, 4])
        node_prediction_argmax[:, :, idx_node_arg_max] = 1

        # get discretized node-prediction and append it to list
        node_prediction_argmax_squeezed = node_prediction_argmax.squeeze(
            0).to(device)
        x_list.append(node_prediction_argmax_squeezed)

        # reset and update input for next iteration
        x_step = torch.zeros((test_batch_size, 1, max_prev_node * edge_feature_dims + node_feature_dims),
                             requires_grad=False).to(device)
        x_step[:, :, :node_feature_dims] = node_prediction_argmax.data
        # .data: we only want to get the contenet of the tensor

        # init Edge/Abs lvl
        hidden_null_1 = torch.zeros(
            (rnn.num_layers - 1, h.size(0), h.size(2)), requires_grad=True).to(device)
        # hidden_null_2 = torch.zeros((rnn.num_layers - 1, h.size(0), h.size(2)), requires_grad=True).to(device)
        h_to_pass = h.permute(1, 0, 2).to(device)
        output.hidden = torch.cat((h_to_pass, hidden_null_1), dim=0).to(device)
        # absence_net.hidden = torch.cat((h_to_pass, hidden_null_2), dim=0).to(device)

        # token Edge lvl - randn best result
        output_x_step = torch.randn(
            test_batch_size, 1, 5, requires_grad=False).to(device)
        # token abs lvl - 0s as SOS
        # abs_x_step = torch.zeros(test_batch_size, 1, 5, requires_grad=False).to(device)

        # Edge/Abs RNN for-loop
        edge_rnn_step = 0
        # this list is used to create edg_idx
        idx = [k for k in range(i, -1, -1)]
        for j in range(min(max_prev_node, i + 1)):
            # prediction for each and every prev node
            output_y_pred_step_out = output(output_x_step)
            # abs_ = absence_net(abs_x_step)
            # abs_ = F.sigmoid(abs_)
            # abs_ = output_y_pred_step_out[:,:,0].item()

            idx_edge_arg_max = torch.argmax(
                output_y_pred_step_out, dim=-1).item()
            output_x_step_argmax = torch.zeros_like(
                output_y_pred_step_out).to(device)  # torch.Size([1, 1, 4])
            output_x_step_argmax[:, :, idx_edge_arg_max] = 1

            # # discretization of abs prediction
            # if abs_.item() >= 1.:
            #     temp_0 = torch.zeros_like(output_x_step_argmax, requires_grad=True).to(device)
            #     output_x_step_argmax = torch.mul(output_x_step_argmax,
            #                                      temp_0)  # this kills gradients, thus params not updated!
            #     t = torch.Tensor([1]).to(device)
            #     abs_out = (abs_ >= t).float()
            #     output_x_step_argmax = torch.cat((abs_out, output_x_step_argmax), dim=-1)
            # else:
            #     t = torch.Tensor([1]).to(device)
            #     abs_out = (abs_ > t).float()
            #     output_x_step_argmax = torch.cat((abs_out, output_x_step_argmax), dim=-1)

            # # reset and update input for next iteration
            # output_x_step = output_x_step_argmax  # abs_out.data
            # abs_x_step = output_x_step_argmax

            if torch.argmax(output_x_step_argmax, dim=-1) != 0:
                if i + 1 <= max_prev_node:
                    # select [1:]
                    idx_select = torch.tensor(
                        [1, 2, 3, 4], dtype=torch.long, requires_grad=False).to(device)
                    edge_to_append = torch.index_select(
                        output_x_step_argmax, dim=2, index=idx_select).to(device)

                    # Duplicate
                    edges_to_append_doubled = torch.cat((edge_to_append.squeeze(), edge_to_append.squeeze()), dim=0).to(
                        device)

                    # reshape to (2x4)
                    edges_to_append_resh = torch.reshape(
                        edges_to_append_doubled, (2, 4)).to(device)

                    # Append to edg_attr_list
                    edg_attr_list.append(edges_to_append_resh)

                    # Edge_index creation
                    edg_idx_list.append(torch.tensor(
                        [i + 1, idx[j]], requires_grad=False))
                    edg_idx_list.append(torch.tensor(
                        [idx[j], i + 1], requires_grad=False))

            # Define next time-step input
            x_step[:, :, 4 * j + node_feature_dims + j: 4 * (j + 1) + node_feature_dims + (
                j + 1)] = output_x_step_argmax.data
            edge_rnn_step = j

        node_to_break, edges_to_break = torch.split(
            x_step, [node_feature_dims, max_prev_node * 5], dim=2)
        edges_to_break_temp = torch.reshape(
            edges_to_break, (edges_to_break.shape[0], max_prev_node, 5)).to(device)
        edges_to_break_uptillnow = edges_to_break_temp[0,
                                                       :edge_rnn_step + 1, :].to(device)
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
        # Edge_idx can be non-differentiable
        edge_idx_temp = torch.stack(edg_idx_list).to(device)
        edge_idx = torch.transpose(edge_idx_temp, 0, 1).to(device)

        # Stack on edge_attributes
        edge_attr_ = torch.stack(edg_attr_list, dim=0).to(device)

        # Reshape to (2*E,4)
        edge_attr = torch.reshape(edge_attr_, (edge_attr_.shape[0] * edge_attr_.shape[1], edge_attr_.shape[-1])).to(
            device)

    else:
        # print('disconnected nodes case')
        data = Data(x=x.to(torch.float32).to(device))
        return data

    data = Data(x=x.to(torch.float32).to(device), edge_index=edge_idx.to(torch.long).to(device),
                edge_attr=edge_attr.to(torch.float32).to(device))

    return data


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


def get_generator():
    # NODE LEVEL AND ABSENCE NET embeddings and hidden sizes
    embedding_size_rnn = 256 *4
    hidden_size_rnn = 256 *4

    # EDGE LEVEL embeddings and hidden sizes
    embedding_size_rnn_output = 128 *4
    hidden_size_rnn_output = 128 *4
    out_edge_level = hidden_size_rnn_output
    num_layers = 1

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
    
    return rnn, output


# guacm_smiles = "/home/nobilm@usi.ch/master_thesis/guacamol/guacamol2_molgpt.smiles"
guacm_smiles = "/home/nobilm@usi.ch/master_thesis/guacamol/testdata.smiles"
guac_mols = mols_from_file(guacm_smiles, True)
atom2num, num2atom, max_num_node = get_atoms_info(guac_mols)
bond2num = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
num2bond = {v: k for k, v in bond2num.items()}

data = rdkit2pyg([guac_mols[0]])  # HERE

device, cuda, train_log, test_log = setup()
max_num_node = 88
node_feature_dims = 12
edge_feature_dims = 5
max_prev_node = max_num_node - 1
LR = 2e-3
wd = 5e-3

train_dataset_loader, _ = create_train_val_dataloaders(
    data, max_num_node, max_prev_node)
node_weights, edge_weights = torch.ones(
    (node_feature_dims)), torch.ones((edge_feature_dims))

rnn, output = get_generator()
optimizer_rnn = torch.optim.AdamW(
    list(rnn.parameters()), lr=LR)  # , weight_decay=wd)
optimizer_output = torch.optim.AdamW(
    list(output.parameters()), lr=LR)  # , weight_decay=wd)

if cuda:
    rnn.to(device)
    output.to(device)

# model.apply(weight_init)
rnn.apply(weight_init)
output.apply(weight_init)
epoch = 1  # starting epoch
max_epoch = 1000
# patience = 100
counter_test = 0

# from torch_lr_finder import LRFinder
# import matplotlib.pyplot as plt
# scheRNN = torch.optim.lr_scheduler.ExponentialLR(optimizer_rnn, 1.1)
# scheOUT = torch.optim.lr_scheduler.ExponentialLR(optimizer_output, 1.1)

scheduler1 = torch.optim.lr_scheduler.OneCycleLR(optimizer_rnn, max_lr=LR, steps_per_epoch=len(train_dataset_loader), epochs=max_epoch)
scheduler2 = torch.optim.lr_scheduler.OneCycleLR(optimizer_output, max_lr=LR, steps_per_epoch=len(train_dataset_loader), epochs=max_epoch)

lrrnn = []
lrout = []
while epoch <= max_epoch:

    loss_this_epoch, loss_edg, loss_nodes = train_rnn_epoch(rnn=rnn, output=output,
                                                            data_loader_=train_dataset_loader,
                                                            optimizer_rnn=optimizer_rnn,
                                                            optimizer_output=optimizer_output,
                                                            node_weights=node_weights,
                                                            edge_weights=edge_weights,
                                                            device=device)
    # scheduler1.step()
    # scheduler2.step()
    epoch += 1
    train_log.info(
        f'Epoch: {epoch}/{max_epoch}, sum of Loss: {loss_this_epoch:.8f}, loss edges {loss_edg:.8f}, loss nodes {loss_nodes:.8f}')
    
    # lrrnn.append(scheRNN.get_last_lr())
    # lrout.append(scheOUT.get_last_lr())
    # scheRNN.step()
    # scheOUT.step()
# plt.figure()
# plt.plot([i for i in range(len(lrrnn))], lrrnn)
# plt.show()
# plt.savefig('foo.png')


# plt.plot([i for i in range(lrout)], lrout)

# ------------------------------------------------------------------------------------------
Ns = [5]#, 60000, 110000, 160000, 210000]

# for each el in list, call f with el
# from utils.data_utils import get_smiles

def generate_mols(N):
    to_draw = []
    for idx in range(N):
        print(f"{idx+1}/{N}", end='\r')
        obs = generate_single_obs(rnn, output, device, test_batch_size=1,
                                    max_num_node=max_num_node,
                                    max_prev_node=max_prev_node)
        to_draw.append(obs)

    smiles_ = pyg2rdkit(to_draw)
    # path = "/home/nobilm@usi.ch/wd/data/generated_smiles/graphRNN_original_thesis_weights/"
    filename = f"original_thesis_weights_all_{epoch}_{N}.smiles"
    smiles_ = [Chem.MolToSmiles(m) for m in smiles_]
    save_smiles(smiles_, ".", filename, "smiles")


for i in Ns: generate_mols(i)

