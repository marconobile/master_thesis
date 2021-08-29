import os
import torch
from supervised_tools.supervised_model import get_generator


def save(epoch, rnn, output, absence_net):
    path = os.getcwd() + "/weights/"  # "/content/gdrive/My Drive/GraphRNN_weights/"
    checkpoint_rnn = {
        'model_state_dict': rnn.state_dict(),
    }
    torch.save(checkpoint_rnn, path + f'nodeRNN_checkpoint_{epoch}.pth')

    checkpoint_output = {
        'model_state_dict': output.state_dict(),
    }
    torch.save(checkpoint_output, path + f'edgeRNN_checkpoint_{epoch}.pth')

    checkpoint_absence = {
        'model_state_dict': absence_net.state_dict(),
    }
    torch.save(checkpoint_absence, path + f'absenceRNN_checkpoint_{epoch}.pth')

    return None


def load_nets(cuda, device, to_be_loaded):  # to_be_loaded = int(last_epoch)
    path = "./weights"
    rnn, output, absence_net = get_generator()
    PATH1 = path + f'/nodeRNN_checkpoint_{to_be_loaded}.pth'
    if cuda:
        checkpoint1 = torch.load(PATH1)
    else:
        checkpoint1 = torch.load(PATH1, map_location='cpu')
    rnn.load_state_dict(checkpoint1['model_state_dict'])
    PATH2 = path + f'/edgeRNN_checkpoint_{to_be_loaded}.pth'
    if cuda:
        checkpoint2 = torch.load(PATH2)
    else:
        checkpoint2 = torch.load(PATH2, map_location='cpu')
    output.load_state_dict(checkpoint2['model_state_dict'])
    PATH3 = path + f'/absenceRNN_checkpoint_{to_be_loaded}.pth'
    if cuda:
        checkpoint3 = torch.load(PATH3)
    else:
        checkpoint3 = torch.load(PATH3, map_location='cpu')
    absence_net.load_state_dict(checkpoint3['model_state_dict'])

    rnn.to(device)
    output.to(device)
    absence_net.to(device)

    rnn.eval()
    output.eval()
    absence_net.eval()

    return rnn, output, absence_net
