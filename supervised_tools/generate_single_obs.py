import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def generate_single_obs(rnn, output, absence_net, device, args, max_num_node, max_prev_node, test_batch_size=1):
    # initialize hidden state randomly (constant during training)
    rnn.hidden = rnn.init_hidden_rand(test_batch_size).to(device)



    # create node level token
    x_step = torch.ones((test_batch_size, 1, max_prev_node * args.edge_feature_dims + args.node_feature_dims),
                        requires_grad=False).to(device)

    # initialize empty lists for Data() object
    x_list = []
    edg_attr_list = []
    edg_idx_list = []

    # Node RNN for-loop
    for i in range(max_num_node):
        h, node_prediction = rnn(x_step)
        node_prediction = F.log_softmax(node_prediction, dim=2)

        # ARG-MAX for nodes
        idx_node_arg_max = torch.argmax(node_prediction, dim=-1).item()
        node_prediction_argmax = torch.zeros_like(node_prediction).to(device)  # torch.Size([1, 1, NF])
        node_prediction_argmax[:, :, idx_node_arg_max] = 1

        # get discretized node-prediction and append it to list
        node_prediction_argmax_squeezed = node_prediction_argmax.squeeze(0).to(device)
        x_list.append(node_prediction_argmax_squeezed)

        # reset and update input for next iteration
        x_step = torch.zeros((test_batch_size, 1, max_prev_node * args.edge_feature_dims + args.node_feature_dims),
                             requires_grad=False).to(device)
        x_step[:, :, :args.node_feature_dims] = node_prediction_argmax.data

        # init Edge/Abs lvl
        hidden_null_1 = torch.zeros((rnn.num_layers - 1, h.size(0), h.size(2)), requires_grad=True).to(device)
        hidden_null_2 = torch.zeros((rnn.num_layers - 1, h.size(0), h.size(2)), requires_grad=True).to(device)
        h_to_pass = h.permute(1, 0, 2).to(device)
        output.hidden = torch.cat((h_to_pass, hidden_null_1), dim=0).to(device)
        absence_net.hidden = torch.cat((h_to_pass, hidden_null_2), dim=0).to(device)

        # token Edge lvl, so I do pass randn as first-step-token in input to the edge-lvl-rnn
        output_x_step = torch.ones(test_batch_size, 1, args.edge_feature_dims, requires_grad=False).to(device)
        # token abs lvl
        abs_x_step = torch.ones(test_batch_size, 1, args.edge_feature_dims, requires_grad=False).to(device)

        # Edge/abs RNN for-loop
        edge_rnn_step = 0
        idx = [k for k in range(i, -1, -1)]  # this list is used to create edg_idx
        for j in range(min(max_prev_node, i + 1)):

            output_y_pred_step_out = output(output_x_step)
            output_y_pred_step_out = F.log_softmax(output_y_pred_step_out, dim=2)

            abs_ = absence_net(abs_x_step)
            abs_ = F.sigmoid(abs_)

            # ARG-MAX for edges
            idx_edge_arg_max = torch.argmax(output_y_pred_step_out, dim=-1).item()
            output_x_step_argmax = torch.zeros_like(output_y_pred_step_out).to(device)  # torch.Size([1, 1, ef])
            output_x_step_argmax[:, :, idx_edge_arg_max] = 1

            # discretization of abs prediction
            if abs_.item() >= 0.5:
                t = torch.Tensor([0.5]).to(device)
                abs_out = (abs_ >= t).float()
                temp_0 = torch.zeros_like(output_x_step_argmax, requires_grad=True).to(device)
                output_x_step_argmax = torch.mul(output_x_step_argmax,
                                                 temp_0)
                output_x_step_argmax = torch.cat((abs_out, output_x_step_argmax), dim=-1)
            else:
                temp_0 = torch.zeros_like(abs_)
                output_x_step_argmax = torch.cat((temp_0, output_x_step_argmax), dim=-1)

            # reset and update input for next iteration
            abs_x_step = output_x_step_argmax
            output_x_step = output_x_step_argmax

            if torch.argmax(output_x_step_argmax, dim=-1) != 0:
                if i + 1 <= max_prev_node:
                    # select [1:]
                    idx_select = torch.range(1, 3, dtype=torch.long, requires_grad=False).to(device)
                    edge_to_append = torch.index_select(output_x_step_argmax, dim=2, index=idx_select).to(device)

                    # Duplicate
                    edges_to_append_doubled = torch.cat((edge_to_append.squeeze(), edge_to_append.squeeze()), dim=0).to(
                        device)

                    # reshape to (2x4)
                    edges_to_append_resh = torch.reshape(edges_to_append_doubled, (2, args.edge_feature_dims - 1)).to(
                        device)

                    # Append to edg_attr_list
                    edg_attr_list.append(edges_to_append_resh)

                    # Edge_index creation
                    edg_idx_list.append(torch.tensor([i + 1, idx[j]], requires_grad=False))
                    edg_idx_list.append(torch.tensor([idx[j], i + 1], requires_grad=False))

            # Define next time-step input
            x_step[:, :, (args.edge_feature_dims - 1) * j + args.node_feature_dims + j: (args.edge_feature_dims - 1) * (
                    j + 1) + args.node_feature_dims + (j + 1)] = output_x_step_argmax.data

            edge_rnn_step = j

        node_to_break, edges_to_break = torch.split(x_step,
                                                    [args.node_feature_dims, max_prev_node * args.edge_feature_dims],
                                                    dim=2)
        edges_to_break_temp = torch.reshape(edges_to_break,
                                            (edges_to_break.shape[0], max_prev_node, args.edge_feature_dims)).to(device)
        edges_to_break_uptill_now = edges_to_break_temp[0, :edge_rnn_step + 1, :].to(device)
        break_ = True
        for row in edges_to_break_uptill_now:
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
        data = Data(x=x.to(torch.float32).to(device))
        return data

    data = Data(x=x.to(torch.float32).to(device), edge_index=edge_idx.to(torch.long).to(device),
                edge_attr=edge_attr.to(torch.float32).to(device))

    return data
