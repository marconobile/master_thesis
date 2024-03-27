import logging
import numpy as np
import random
# from model import get_generator
import torch


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def setup_device(device: int = None):

    if device == None: device = 1
    manualSeed = 123
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    cuda = True if torch.cuda.is_available() else False
    device = torch.device(f"cuda:{device}" if cuda else "cpu")

    if cuda:
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False

    return device, cuda

def setup_loss_loggers():
    '''
    Setup of loggers, cuda and seeds.
    '''
    setup_logger('train_loss_log_new_abs', r'./train_loss_log_new_abs')
    setup_logger('val_loss_log_new_abs', r'./val_loss_log_new_abs')

    train_log = logging.getLogger('train_loss_log_new_abs')
    val_log = logging.getLogger('val_loss_log_new_abs')
    return train_log, val_log


# def load_nets(cuda, device, to_be_loaded):
#     '''
#     to_be_loaded = int(last_epoch)
#     '''
#     path = "./weights"
#     rnn, output = get_generator()
#     PATH1 = path + f'/nodeRNN_checkpoint_{to_be_loaded}.pth'
#     if cuda: checkpoint1 = torch.load(PATH1)
#     else: checkpoint1 = torch.load(PATH1, map_location='cpu')
#     rnn.load_state_dict(checkpoint1['model_state_dict'])
#     PATH2 = path + f'/edgeRNN_checkpoint_{to_be_loaded}.pth'
#     if cuda: checkpoint2 = torch.load(PATH2)
#     else: checkpoint2 = torch.load(PATH2, map_location='cpu')
#     output.load_state_dict(checkpoint2['model_state_dict'])

#     rnn.to(device)
#     output.to(device)
#     rnn.eval()
#     output.eval()
#     return rnn, output
