import torch.nn as nn
from model import get_generator
from supervised_tools.create_train_val_data import create_train_val_dataloaders
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
from rdkit import Chem
from utils.setup import setup_loss_loggers, setup_device
import torch
from utils.data_utils import mols_from_file, rdkit2pyg, pyg2rdkit, save_smiles
from mappings import *
import torch.nn.init as init
import time
for i in range(10):
    torch.cuda.empty_cache()
import os

def generate_file(path, filename):
    '''
    if path does not exist it is created
    if filename must have extension, default: '.txt'
    if file already exists it is overwritten

    args:
        - path directory where to save smiles list as txt
        - filename name of the file. By default it is created a txt file
    '''
    os.makedirs(path, exist_ok=True)
    path_to_file = os.path.join(path, filename)
    filename_ext = os.path.splitext(path_to_file)[-1].lower()
    if not filename_ext:
        path_to_file += '.txt'
    if os.path.isfile(path_to_file):
        try:
            os.remove(path_to_file)
        except OSError:
            raise f"{path_to_file} already existing and could not be removed"

    cmd = f"touch {path_to_file};"
    os.system(cmd)
    return path_to_file


def create_log(path=".", name="log.txt"):
    if not name.endswith(".txt"):
        name += ".txt"
    generate_file(path, name)
    return os.path.join(path, name)


def append_line_to_log(path_to_log, line):
    with open(path_to_log, "a") as log:
        if isinstance(line, list):
            for w in line:
                log.write(str(w) + "\n")
        else:
            log.write(str(line) + "\n")


class TimeCode:
    '''
    Utility class that times code execution.
    Usage: instanciate it when code starts
    call compute compute_ellapsed when you want to evaluate passed time
    '''

    def __init__(self):
        self.start_timer()

    def start_timer(self):
        self.start = time.time()

    def compute_ellapsed(self):
        self.end = time.time()
        print(self.end - self.start)
        return self.end - self.start  # in seconds


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


def train_rnn_epoch(rnn, output, data_loader_, optimizer_rnn, optimizer_output, node_weights, edge_weights):

    rnn.train()
    output.train()
    loss_sum, loss_sum_edges, loss_sum_nodes = 0, 0, 0

    N = len(data_loader_)
    for batch_idx, data in enumerate(data_loader_):

        rnn.zero_grad()
        output.zero_grad()

        loss, edge_loss, node_loss = fit_batch(data, rnn, output, node_weights, edge_weights)

        loss.backward(retain_graph=True)
        optimizer_output.step()
        optimizer_rnn.step()

        loss_sum += loss.item()
        loss_sum_edges += edge_loss.item()
        loss_sum_nodes += node_loss.item()
        print(f'Batch {batch_idx+1}/{N}', end='\r')

    return loss_sum/N, loss_sum_edges/N, loss_sum_nodes/N


@torch.no_grad()
def validate_rnn_epoch(rnn, output, data_loader_, node_weights, edge_weights):
    rnn.eval()
    output.eval()
    loss_sum, loss_sum_edges, loss_sum_nodes = 0, 0, 0
    for batch_idx, data in enumerate(data_loader_):
        loss, edge_loss, node_loss = fit_batch(data, rnn, output, node_weights, edge_weights)
        loss_sum += loss.item()
        loss_sum_edges += edge_loss.item()
        loss_sum_nodes += node_loss.item()

    return loss_sum / (batch_idx + 1), loss_sum_edges / (batch_idx + 1), loss_sum_nodes / (batch_idx + 1)


def fit_batch(data, rnn, output, node_weights, edge_weights):
    with torch.no_grad():
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

        x_unsorted_for_nn = torch.reshape(x_unsorted, (x_unsorted.shape[0], x_unsorted.shape[1], x_unsorted.shape[2] * x_unsorted.shape[3]))
        y_unsorted_for_nn = torch.reshape(y_unsorted, (x_unsorted.shape[0], x_unsorted.shape[1], x_unsorted.shape[2] * x_unsorted.shape[3]))

        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted_for_nn.size(0))

        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist() # ([bs, max_seq_l, 8*4])
        x = torch.index_select(x_unsorted_for_nn, 0, sort_index)
        y = torch.index_select(y_unsorted_for_nn, 0, sort_index) # ([bs, max_seq_l, node_f])
        x_nodes = torch.index_select(x_nodes_unsorted, 0, sort_index)
        y_nodes = torch.index_select(y_nodes_unsorted, 0, sort_index).to(device)  # NODE TARGETS
        y_reshape = torch.nn.utils.rnn.pack_padded_sequence(y, y_len, batch_first=True).data
        idx = torch.LongTensor([i for i in range(y_reshape.size(0) - 1, -1, -1)])
        # inverts the rows order of y_reshape
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0), max_prev_node, edge_feature_dims)
        output_x = torch.cat((torch.ones(y_reshape.size(0), 1, edge_feature_dims), y_reshape[:, 0:-1, :]), dim=1).to(device)
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

        x = torch.cat((x_nodes, x), 2).to(device)  # INPUT FOR NODE LVL
        output_y.to(device)

    # OUTPUTS
    h, node_prediction = rnn(x, pack=True, input_len=y_len)

    h = torch.nn.utils.rnn.pack_padded_sequence(h, y_len, batch_first=True).data  # get packed hidden vector
    idx = torch.LongTensor([i for i in range(h.size(0) - 1, -1, -1)]).to(device)
    h = h.index_select(0, idx)

    hidden_null = torch.zeros(rnn.num_layers - 1, h.size(0), h.size(1)).to(device) # num_layers, batch_size, hidden_size
    output.hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null), dim=0)
    y_pred = output(output_x, pack=True, input_len=output_y_len)

    # OUTPUT LAYERS
    y_pred = F.log_softmax(y_pred, dim=2)
    node_prediction = F.log_softmax(node_prediction, dim=2)

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

    # permutations of predicted edg/nodes
    y_pred = y_pred.permute(0, 2, 1)
    node_prediction = node_prediction.permute(0, 2, 1)

    # permutations of real edg/nodes
    output_y = output_y.permute(0, 2, 1)
    y_nodes = y_nodes.permute(0, 2, 1)

    _, indices_edges = torch.max(output_y, 1)
    _, indices_nodes = torch.max(y_nodes, 1)

    edge_loss = F.nll_loss(y_pred, indices_edges.to(device), reduction='mean', weight=edge_weights.to(torch.float32).to(device))
    node_loss = F.nll_loss(node_prediction, indices_nodes.to(device), reduction='mean', weight=node_weights.to(torch.float32).to(device))

    edge_loss *= 1000
    node_loss *= 1000

    loss = edge_loss + node_loss
    return loss, edge_loss, node_loss


@torch.no_grad()
def generate_single_obs(rnn, output, device, max_num_node, max_prev_node, test_batch_size=1):
    rnn.eval()
    output.eval()
    if MEMORIZATION:
        rnn.hidden = rnn.init_hidden(test_batch_size).to(device)  # rand
    else:
        rnn.hidden = rnn.init_hidden_rand(test_batch_size).to(device)  # rand

    x_step = torch.ones((test_batch_size, 1, max_prev_node * edge_feature_dims + node_feature_dims)).to(device) # create node level token
    x_list, edg_attr_list, edg_idx_list = [], [], [] # initialize empty lists for Data() object

    for i in range(max_num_node): # Node RNN for-loop
        h, node_prediction = rnn(x_step)

        node_prediction_argmax = F.one_hot(node_prediction.argmax(), num_classes=node_feature_dims)
        x_list.append(node_prediction_argmax)

        # reset and update input for next iteration
        x_step = torch.zeros((test_batch_size, 1, max_prev_node * edge_feature_dims + node_feature_dims)).to(device)
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
        # Edge_idx can be non-differentiable
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
def generate_mols(N):
    to_draw = []
    for idx in range(N):
        print(f"{idx+1}/{N}", end='\r')
        try:
            obs = generate_single_obs(rnn, output, device, test_batch_size=1,
                                        max_num_node=max_num_node,
                                        max_prev_node=max_prev_node)
            to_draw.append(obs)
        except:
            pass

    smiles_ = pyg2rdkit(to_draw)
    # path = "/home/nobilm@usi.ch/wd/data/generated_smiles/graphRNN_original_thesis_weights/"
    filename = f"original_thesis_weights_all_{epoch}_{N}_new_abs_1000.smiles"
    smiles_ = [Chem.MolToSmiles(m) for m in smiles_]
    save_smiles(smiles_, ".", filename, "smiles")


# --- GET DATA ---
MEMORIZATION = False
train_data = rdkit2pyg(mols_from_file("/home/nobilm@usi.ch/master_thesis/guacamol/30k_test.smiles", True)); valid_data = train_data

# train_data = rdkit2pyg(mols_from_file("./guacamol/guacamol_v1_train.smiles", True))
# valid_data = rdkit2pyg(mols_from_file("./guacamol/guacamol_v1_valid.smiles", True))

# MEMORIZATION = True
# obs = train_guac_mols[6343]
# # with old weights: 1 2 6343 OK
# # with new weights: 1 2 OK
# print(Chem.MolToSmiles(obs))
# train_data = rdkit2pyg([obs])
# valid_data = train_data


#! --- GET WEIGHTS ---
# nweights = {
#     'C':    0.03238897867833534,
#     'Br':   14.044943820224718,
#     'N':    0.21620219229022983,
#     'O':    0.2177273617975571,
#     'S':    1.6680567139282736,
#     'Cl':   2.872737719046251,
#     'F':    1.754693805930865,
#     'P':    37.735849056603776,
#     'I':    100.0,
#     'B':    416.6666666666667,
#     'Si':   454.54545454545456,
#     'Se':   833.3333333333334
# }
# bweights = {
#     BT.SINGLE:      4.663287337775892,
#     BT.AROMATIC:    4.77780803722868,
#     BT.DOUBLE:      34.74514436607484,
#     BT.TRIPLE:      969.9321047526673
# }

nweights = {
        'C': 0.03187551336784888,
    'Br': 2.7110419759590325,
        'N': 0.19573304627602645,
        'O': 0.1969863033428892,
        'S': 0.9813503849220039,
    'Cl':1.353961677879625,
    'F': 1.0133062952053178,
    'P':  3.656765503466812,
    'I':   4.61512051684126,
    'B':   6.034683666227958,
    'Si':  6.121495502161354,
    'Se':  6.7266330027636645,
}

bweights = {
    BT.SINGLE:  1.7340045253422367,
    BT.AROMATIC:  1.75402437844415,
    BT.DOUBLE:  3.576414437987407,
    BT.TRIPLE:  6.878256542831836,
}


nweights_list = [nweights[k] for k in atom2num]
bweights_list = [bweights[k] for k in bond2num]
# bweights_list.insert(0, 1500)
bweights_list.insert(0, 1.7340045253422367+6.878256542831836)
node_weights = torch.tensor(nweights_list)
edge_weights = torch.tensor(bweights_list)


#! --- SET UP EXPERIMENT ---
LRrnn, LRout = 1e-5, 1e-5
wd = 10e-6
bs_train = 64
epoch, max_epoch = 1, 20000
device, cuda = setup_device()
train_log, val_log = setup_loss_loggers()
train_dataset_loader, val_dataset_loader = create_train_val_dataloaders(train_data, valid_data, max_num_node, max_prev_node, bs_train)

rnn, output = get_generator()
rnn.apply(weight_init)
output.apply(weight_init)
rnn.ad_hoc_init()
output.ad_hoc_init()
optimizer_rnn = torch.optim.RMSprop(list(rnn.parameters()), lr=LRrnn, eps=1e-5, weight_decay=wd)
optimizer_output = torch.optim.RMSprop(list(output.parameters()), lr=LRout, eps=1e-5, weight_decay=wd)
scheduler_rnn = None # torch.optim.lr_scheduler.OneCycleLR(optimizer_rnn, max_lr=LRrnn, steps_per_epoch=len(train_dataset_loader), epochs=max_epoch)
scheduler_output = None # torch.optim.lr_scheduler.OneCycleLR(optimizer_output, max_lr=LRout, steps_per_epoch=len(train_dataset_loader), epochs=max_epoch)


print("start training")
VALIDATION = False
min_val_loss_nodes = torch.inf
min_val_loss_edges = torch.inf
min_val_epoch = 0
Ns = [100]#, 60000, 110000, 160000, 210000]
gen_every = 20


timer = TimeCode()
try:
    while epoch <= max_epoch:
        loss_this_epoch, loss_edg, loss_nodes = train_rnn_epoch(rnn=rnn, output=output,
                                                                data_loader_=train_dataset_loader,
                                                                optimizer_rnn=optimizer_rnn, optimizer_output=optimizer_output,
                                                                node_weights=node_weights, edge_weights=edge_weights)
        if scheduler_rnn != None: scheduler_rnn.step()
        if scheduler_output != None: scheduler_output.step()
        if epoch % gen_every == 0:
            for i in Ns: generate_mols(i)
        if MEMORIZATION:
            if epoch % 200:
                train_log.info(f'Epoch: {epoch}/{max_epoch}, sum of Loss: {loss_this_epoch:.8f}, loss edges {loss_edg:.8f}, loss nodes {loss_nodes:.8f}')
        else:
            train_log.info(f'Epoch: {epoch}/{max_epoch}, sum of Loss: {loss_this_epoch:.8f}, loss edges {loss_edg:.8f}, loss nodes {loss_nodes:.8f}')
        if VALIDATION:
            if epoch % 5:
                val_loss, loss_edg, loss_nodes = validate_rnn_epoch(rnn, output, val_dataset_loader, node_weights, edge_weights)
                val_log.info(f'Epoch: {epoch}/{max_epoch}, sum of Loss: {val_loss:.8f}, loss edges {loss_edg:.8f}, loss nodes {loss_nodes:.8f}')

                if loss_nodes <= min_val_loss_nodes and loss_edg<=min_val_loss_edges:
                    min_val_epoch = epoch
                    min_val_loss_nodes = loss_nodes
                    min_val_loss_edges = loss_edg
                    rnn.save(epoch)
                    output.save(epoch)
        epoch += 1
except KeyboardInterrupt:
    print("kyeboard interrupt!")

timer.compute_ellapsed()
print(f"min val loss at epoch: {min_val_epoch}, nodes {min_val_loss_nodes}, edges {min_val_loss_edges}")
Ns = [10]#, 60000, 110000, 160000, 210000]
for i in Ns: generate_mols(i)

# print(Chem.MolToSmiles(obs))
