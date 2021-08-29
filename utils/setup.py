import logging
import numpy as np
import random
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


def setup():
    '''
    Setup of loggers, cuda and seeds.
    '''

    setup_logger('train_loss_log', r'../train_loss_log')
    setup_logger('test_loss_log', r'../test_loss_log')

    train_log = logging.getLogger('train_loss_log')
    test_log = logging.getLogger('test_loss_log')

    manualSeed = 23742
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:0" if cuda else "cpu")

    if cuda:
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return device, cuda, train_log, test_log


def unsupervised_setup():

    setup_logger('critic_loss_log', r'./critic_loss_log')
    critic_loss_log = logging.getLogger('critic_loss_log')
    setup_logger('generator_loss_log', r'./generator_loss_log')
    generator_loss_log = logging.getLogger('generator_loss_log')

    return critic_loss_log, generator_loss_log