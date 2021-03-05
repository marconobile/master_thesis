import torch


def check_params(rnn, output, weights_log):
    # execute after .step()
    node_lvl_after = {}
    ii = 0
    for module in rnn._modules.keys():
        if module != 'relu':
            if (type(rnn._modules[module]) == torch.nn.modules.container.Sequential):
                for _, lin_layer in enumerate(rnn._modules[module]._modules.keys()):
                    if lin_layer != 'relu':
                        for param_ in rnn._modules[module]._modules[lin_layer]._parameters.keys():
                            c = rnn._modules[module]._modules[lin_layer]._parameters[param_].requires_grad
                            a1 = torch.norm(rnn._modules[module]._modules[lin_layer]._parameters[param_].data)
                            a = torch.norm(rnn._modules[module]._modules[lin_layer]._parameters[param_].grad.data)
                            node_lvl_after[ii] = (module, param_, c, a1, a)
                            ii += 1

            else:
                for param in rnn._modules[module]._parameters.keys():
                    c = rnn._modules[module]._parameters[param].requires_grad
                    a1 = torch.norm(rnn._modules[module]._parameters[param].data)
                    a = torch.norm(rnn._modules[module]._parameters[param].grad.data)
                    node_lvl_after[ii] = (module, param, c, a1, a)
                    ii += 1

    for _, value in node_lvl_after.items():
        weights_log.info(
            f'NODE lvl, {value[0]} module, {value[1]} param requires_grad={value[2]}, norm({value[1]})={value[3]}, norm GRAD({value[1]})={value[4]}')

    weights_log.info('###' * 10)

    jj = 0
    edge_lvl_after = {}
    for module in output._modules.keys():
        if module != 'relu':
            if (type(output._modules[module]) == torch.nn.modules.container.Sequential):
                for _, lin_layer in enumerate(output._modules[module]._modules.keys()):
                    if lin_layer != 'relu':
                        for param_ in output._modules[module]._modules[lin_layer]._parameters.keys():
                            c = output._modules[module]._modules[lin_layer]._parameters[param_].requires_grad
                            a1 = torch.norm(output._modules[module]._modules[lin_layer]._parameters[param_].data)
                            a = torch.norm(output._modules[module]._modules[lin_layer]._parameters[param_].grad.data)
                            edge_lvl_after[jj] = (module, param_, c, a1, a)
                            jj += 1
            else:
                for _, param in enumerate(output._modules[module]._parameters.keys()):
                    c = output._modules[module]._parameters[param].requires_grad
                    a1 = torch.norm(output._modules[module]._parameters[param].data)
                    a = torch.norm(output._modules[module]._parameters[param].grad.data)
                    edge_lvl_after[jj] = (module, param, c, a1, a)
                    jj += 1

    for _, value in edge_lvl_after.items():
        weights_log.info(
            f'EDGE lvl, {value[0]} module, {value[1]} param requires_grad={value[2]}, norm({value[1]})={value[3]}, norm GRAD({value[1]})={value[4]}')

    weights_log.info('###' * 10 + "\n")


def freeze_net(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

