from create_train_val_data import create_train_val_dataloaders_geometric
from save_load_model import load_for_gpu, load_reward_net  # , load_for_CPU
import torch
from disc_model import Critic#, Reward_Net_Joint  , Reward_Net_Single
from get_generator import get_generator
from reward_structure import drop_non_sanitizables_get_mol, set_reward
import train
import wgan_train



def gan_generation(dataset, device, cuda, train_log, qm9_smiles, args, weights_log):

    if args.reward_net:
        dataset, rdkit_mols = drop_non_sanitizables_get_mol(dataset)  # for label extraction
        dataset = set_reward(dataset, rdkit_mols, reward_type='valid')  # dataset with labels
        print("Dataset len after dropping non santizable mols from input data:", len(dataset))

    # original data loaded in dataloader, nb if with iso nodes already dropped, also smiles of QM9 already computed
    train_dataset_loader, test_dataset_loader = create_train_val_dataloaders_geometric(dataset)

    if args.pretrained:
        rnn, output, absence_net = load_for_gpu(300)
        print('ALL Networks loaded successfully')
    else:
        if args.gan or args.wgan:
            _, _, absence_net = load_for_gpu(300)
            print('Absence Net ONLY loaded successfully')

    critic = Critic(args=args)
    rnn, output, _ = get_generator()

    if args.reward_net:
        if args.reward_type == 'druglikeness' or args.reward_type == 'synthesizability' or args.reward_type == 'solubility' or args.reward_type == "valid":
            reward_net = Reward_Net_Single()
            print("Reward net for a single metric initialized")
        elif args.reward_type == 'joint':
            if args.reward_net_pretrain:
                reward_net = load_reward_net(9999)
                print("Reward net for joint metrics LOADED")
            else:
                reward_net = Reward_Net_Joint()
                print("Reward net for joint metrics initialized")

    LR = 1e-5

    print('Networks skeletons:')
    print(rnn)
    print('##' * 20)
    print(output)
    print('##' * 20)
    print(absence_net)
    print('##' * 20)
    print(critic)
    if args.reward_net:
        print('##' * 20)
        print(reward_net)
        reward_net.to(device)
        if args.wgan:
            optimizer_reward_net = torch.optim.RMSprop(list(reward_net.parameters()), lr=LR)
        elif args.gan:
            optimizer_reward_net = torch.optim.Adam(list(reward_net.parameters()), lr=1e-4, weight_decay=5e-5)

    if cuda:
        rnn.to(device)
        output.to(device)
        absence_net.to(device)
        critic.to(device)

    if args.wgan:
        optimizer_rnn = torch.optim.RMSprop(list(rnn.parameters()), lr=LR)
        optimizer_output = torch.optim.RMSprop(list(output.parameters()), lr=LR)
        optimizer_critic = torch.optim.RMSprop(list(critic.parameters()), lr=LR)
    elif args.gan:
        optimizer_rnn = torch.optim.Adam(list(rnn.parameters()), lr=1e-4, weight_decay=5e-5)
        optimizer_output = torch.optim.Adam(list(output.parameters()), lr=1e-4, weight_decay=5e-5)
        optimizer_critic = torch.optim.Adam(list(critic.parameters()), lr=1e-4, weight_decay=5e-5)

    max_num_of_epochs = 150

    if args.gan:
        train.train_gan(rnn, output, absence_net, critic,
                        train_dataset_loader,
                        optimizer_rnn, optimizer_output, optimizer_critic,
                        max_num_of_epochs,
                        device, train_log, qm9_smiles, weights_log, args)
                        #reward_net, optimizer_reward_net)
    elif args.wgan:
        wgan_train.train_gan(rnn, output, absence_net, critic,
                             train_dataset_loader,
                             optimizer_rnn, optimizer_output, optimizer_critic,
                             max_num_of_epochs,
                             device, train_log, qm9_smiles, weights_log, args)#,
                             #reward_net, optimizer_reward_net)
