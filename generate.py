import os
from supervised_tools.save_load_model import load_nets
from utils.data_utils import get_smiles
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

device = "cuda:0"
epoch = 2500
rnn, output, absence_net = load_nets(cuda=True, device=device, to_be_loaded=epoch)  # to_be_loaded = int(last_epoch)

device = "cpu"
rnn.to(device)
output.to(device)
absence_net.to(device)

edge_feature_dims = 5
node_feature_dims = 9
max_num_node = 37
max_prev_node = max_num_node -1

def generate_single_obs(rnn, output, absence_net, device, max_num_node, max_prev_node, test_batch_size=1):
    # initialize hidden state
    rnn.hidden = rnn.init_hidden_rand(test_batch_size).to(device)
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
        node_prediction_argmax = torch.zeros_like(node_prediction).to(device)  # torch.Size([1, 1, 4])
        node_prediction_argmax[:, :, idx_node_arg_max] = 1

        # get discretized node-prediction and append it to list
        node_prediction_argmax_squeezed = node_prediction_argmax.squeeze(0).to(device)
        x_list.append(node_prediction_argmax_squeezed)

        # reset and update input for next iteration
        x_step = torch.zeros((test_batch_size, 1, max_prev_node * edge_feature_dims + node_feature_dims),
                             requires_grad=False).to(device)
        x_step[:, :, :node_feature_dims] = node_prediction_argmax.data
        # .data: we only want to get the contenet of the tensor

        # init Edge/Abs lvl
        hidden_null_1 = torch.zeros((rnn.num_layers - 1, h.size(0), h.size(2)), requires_grad=True).to(device)
        hidden_null_2 = torch.zeros((rnn.num_layers - 1, h.size(0), h.size(2)), requires_grad=True).to(device)
        h_to_pass = h.permute(1, 0, 2).to(device)
        output.hidden = torch.cat((h_to_pass, hidden_null_1), dim=0).to(device)
        absence_net.hidden = torch.cat((h_to_pass, hidden_null_2), dim=0).to(device)

        # token Edge lvl - randn best result
        output_x_step = torch.randn(test_batch_size, 1, 5, requires_grad=False).to(device)
        # token abs lvl - 0s as SOS
        abs_x_step = torch.zeros(test_batch_size, 1, 5, requires_grad=False).to(device)

        # Edge/Abs RNN for-loop
        edge_rnn_step = 0
        idx = [k for k in range(i, -1, -1)]  # this list is used to create edg_idx
        for j in range(min(max_prev_node, i + 1)):
            # prediction for each and every prev node
            output_y_pred_step_out = output(output_x_step)
            abs_ = absence_net(abs_x_step)
            abs_ = F.sigmoid(abs_)

            idx_edge_arg_max = torch.argmax(output_y_pred_step_out, dim=-1).item()
            output_x_step_argmax = torch.zeros_like(output_y_pred_step_out).to(device)  # torch.Size([1, 1, 4])
            output_x_step_argmax[:, :, idx_edge_arg_max] = 1

            # discretization of abs prediction
            if abs_.item() >= 0.5:
                temp_0 = torch.zeros_like(output_x_step_argmax, requires_grad=True).to(device)
                output_x_step_argmax = torch.mul(output_x_step_argmax,
                                                 temp_0)  # this kills gradients, thus params not updated!
                t = torch.Tensor([0.5]).to(device)
                abs_out = (abs_ >= t).float()
                output_x_step_argmax = torch.cat((abs_out, output_x_step_argmax), dim=-1)
            else:
                t = torch.Tensor([0.5]).to(device)
                abs_out = (abs_ > t).float()
                output_x_step_argmax = torch.cat((abs_out, output_x_step_argmax), dim=-1)

            # reset and update input for next iteration
            output_x_step = output_x_step_argmax  # abs_out.data
            abs_x_step = output_x_step_argmax

            if torch.argmax(output_x_step_argmax, dim=-1) != 0:
                if i + 1 <= max_prev_node:
                    # select [1:]
                    idx_select = torch.tensor([1, 2, 3, 4], dtype=torch.long, requires_grad=False).to(device)
                    edge_to_append = torch.index_select(output_x_step_argmax, dim=2, index=idx_select).to(device)

                    # Duplicate
                    edges_to_append_doubled = torch.cat((edge_to_append.squeeze(), edge_to_append.squeeze()), dim=0).to(
                        device)

                    # reshape to (2x4)
                    edges_to_append_resh = torch.reshape(edges_to_append_doubled, (2, 4)).to(device)

                    # Append to edg_attr_list
                    edg_attr_list.append(edges_to_append_resh)

                    # Edge_index creation
                    edg_idx_list.append(torch.tensor([i + 1, idx[j]], requires_grad=False))
                    edg_idx_list.append(torch.tensor([idx[j], i + 1], requires_grad=False))

            # Define next time-step input
            x_step[:, :, 4 * j + node_feature_dims + j: 4 * (j + 1) + node_feature_dims + (
                        j + 1)] = output_x_step_argmax.data
            edge_rnn_step = j

        node_to_break, edges_to_break = torch.split(x_step, [node_feature_dims, max_prev_node * 5], dim=2)
        edges_to_break_temp = torch.reshape(edges_to_break, (edges_to_break.shape[0], max_prev_node, 5)).to(device)
        edges_to_break_uptillnow = edges_to_break_temp[0, :edge_rnn_step + 1, :].to(device)
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


# ------------------------------------------------------------------------------------------
Ns = [10000, 60000, 110000, 160000, 210000]

# for each el in list, call f with el

def generate_mols(N):
    to_draw = []
    for idx in range(N):
        print(f"{idx+1}/{N}", end='\r')
        obs = generate_single_obs(rnn, output, absence_net, device, test_batch_size=1,
                                    max_num_node=max_num_node,
                                    max_prev_node=max_prev_node)
        to_draw.append(obs)

    smiles_ = get_smiles(to_draw)
    path = "/home/nobilm@usi.ch/wd/data/generated_smiles/graphRNN_original_thesis_weights/"
    filename = f"original_thesis_weights_all_{epoch}_{N}.smiles"
    save_smiles(smiles_, path, filename, "smiles")


for i in Ns: generate_mols(i)


# # import torch.multiprocessing as mp
# # with Pool(processes=5) as P: P.map(generate_mols, Ns )


# import torch.multiprocessing as mp

# # Number of processes
# num_processes = 5
# # Share the model's memory to allow it to be accessed by multiple processes

# rnn.share_memory()
# output.share_memory()
# absence_net.share_memory()

# # Create a list of processes and start each process with the train function
# processes = []
# for rank in range(num_processes):
#     p = mp.Process(target=generate_mols, args=(Ns[rank],), name=f'Process-{rank}')
#     p.start()
#     processes.append(p)
#     print(f'Started {p.name}')

# # Wait for all processes to finish
# for p in processes:
#     p.join()
#     print(f'Finished {p.name}')