#!/home/marcon/miniconda3/bin/python
import warnings

warnings.filterwarnings("ignore")

import os
import pickle
from utils.setup import setup
import torch.nn.functional as F

from args import Args
from utils.data_utils import get_smiles
from data.zinc_dataloader import load_zinc
from random import shuffle
# from supervised_tools.supervised_training import supervised_training
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
        try:
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
        except:
            pass

    return mols_



from main import pyg2rdkit
from supervised_tools.create_train_val_data import create_train_val_dataloaders, get_log_weights
import torch

from utils.data_utils import mols_smiles_plots, mols_txt
from supervised_tools.save_load_model import save
from supervised_tools.supervised_train_loop import train_rnn_epoch, test_rnn_single_epoch
from supervised_tools.generate_single_obs import generate_single_obs
from supervised_tools.supervised_model import get_generator
# from new_model import get_generator
# import pickle


def supervised_training(dataset, device, cuda, train_log, test_log, qm9_smiles, test_set):
    '''
    dataset: list of pyg Data objects
    '''

    print("Starting supervised training...")
    
    max_num_node = 88
    max_prev_node = max_num_node - 1
    LR = 5e-2
    wd = 5e-3

    train_dataset_loader, _ = create_train_val_dataloaders(dataset, max_num_node,
                                                           max_prev_node)

    if test_set:
        test_dataset_loader, _ = create_train_val_dataloaders(test_set, max_num_node,
                                                              max_prev_node)

    # get weights for NLL
    # node_weights, edge_weights = get_log_weights(dataset, args)
    # edge_weights[0] = 100  # ad-hoc choice

    node_feature_dims = 12
    edge_feature_dims = 5
    node_weights = torch.ones((node_feature_dims))
    edge_weights = torch.ones((edge_feature_dims))

    print("#N. batches in train_dataset_loader: ", len(train_dataset_loader))
    # print('Node_weights', node_weights, '\nEdge_weights', edge_weights)

    # get models:
    rnn, output, absence_net = get_generator()
    optimizer_abs_net = torch.optim.RMSprop(list(absence_net.parameters()), lr=LR)# , weight_decay=wd)
    optimizer_rnn = torch.optim.RMSprop(list(rnn.parameters()), lr=LR)# , weight_decay=wd)
    optimizer_output = torch.optim.RMSprop(list(output.parameters()), lr=LR)# , weight_decay=wd)

    print('Networks skeletons:')
    print(rnn)
    print('##' * 20)
    print(output)
    print('##' * 20)
    print(absence_net)

    if cuda:
        rnn.to(device)
        output.to(device)
        absence_net.to(device)

    print('Nets structures loaded correctly! Training... ')

    epoch = 1  # starting epoch
    max_epoch = 100
    patience = 100
    counter_test = 0

    while epoch <= max_epoch:

        loss_this_epoch, loss_edg, loss_nodes, loss_abs = train_rnn_epoch(rnn=rnn, output=output,
                                                                          data_loader_=train_dataset_loader,
                                                                          optimizer_rnn=optimizer_rnn,
                                                                          optimizer_output=optimizer_output,
                                                                          node_weights=node_weights,
                                                                          edge_weights=edge_weights,
                                                                          device=device,
                                                                          absence_net=absence_net,
                                                                          absence_net_opt=optimizer_abs_net)

        train_log.info(
            f'Epoch: {epoch}/{max_epoch}, sum of Loss: {loss_this_epoch:.8f}, loss edges {loss_edg:.8f}, loss nodes {loss_nodes:.8f} , loss_abs {loss_abs:.8f}')

        # if test_set and epoch % patience == 0:
        #     test_loss, loss_edg_test, loss_nodes_test, loss_abs = test_rnn_single_epoch(rnn=rnn,
        #                                                                                 output=output,
        #                                                                                 data_loader_=test_dataset_loader,
        #                                                                                 node_weights=node_weights,
        #                                                                                 edge_weights=edge_weights,
        #                                                                                 device=device,
        #                                                                                 absence_net=absence_net)
        #     test_log.info(
        #         f'Evaluation step number {counter_test + 1} (epoch {epoch}), total loss value: {test_loss:.8f}, loss edges {loss_edg_test:.8f}, loss nodes {loss_nodes_test:.8f} , loss_abs {loss_abs:.8f}')
        #     counter_test += 1

    #     if epoch % patience == 0:
    #         save(epoch=epoch, rnn=rnn, output=output, absence_net=absence_net)
    #         print(f'Model saved at epoch {epoch}!')

    #         rnn.eval()
    #         output.eval()
    #         absence_net.eval()

    #         n_of_graph_to_be_generated = 1600
    #         print(f'Generating {n_of_graph_to_be_generated} graphs for epoch', epoch)

    #         to_draw = []
    #         for _ in range(n_of_graph_to_be_generated):
    #             obs = generate_single_obs(rnn, output, absence_net, device, test_batch_size=1,
    #                                       max_num_node=max_num_node,
    #                                       max_prev_node=max_prev_node)
    #             to_draw.append(obs)

    #         rdkit_mols, mols_smiles = mols_smiles_plots(to_draw, './figures/FIG_epoch_' + str(epoch))
    #         mols_txt(epoch, rdkit_mols, mols_smiles, qm9_smiles)

        epoch += 1

    # save(epoch=epoch, rnn=rnn, output=output, absence_net=absence_net)

    print(f'Model saved after LAST epoch {epoch}!')
    print('Writer closed, networks trained')
    print('Script END')
    # return None

    # ------------------------------------------------------------------------------------------


    def save_smiles(smiles, path, filename, ext='.txt'):
        import os
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

    Ns = [5]#, 60000, 110000, 160000, 210000]

    # for each el in list, call f with el
    # from utils.data_utils import get_smiles

    def generate_mols(N):
        to_draw = []
        for idx in range(N):
            print(f"{idx+1}/{N}", end='\r')
            obs = generate_single_obs(rnn, output, absence_net, device, test_batch_size=1,
                                        max_num_node=max_num_node,
                                        max_prev_node=max_prev_node)
            to_draw.append(obs)        

        smiles_ = pyg2rdkit(to_draw)
        # path = "/home/nobilm@usi.ch/wd/data/generated_smiles/graphRNN_original_thesis_weights/"
        from rdkit import Chem
        filename = f"original_thesis_weights_all_{epoch}_{N}.smiles"
        smiles_ = [Chem.MolToSmiles(m) for m in smiles_]
        save_smiles(smiles_, ".", filename, "smiles")


    for i in Ns: generate_mols(i)


if __name__ == "__main__":

    args = Args()
    device, cuda, train_log, test_log = setup()
    cwd = os.getcwd()

    # if args.ZINC_dataset == True:
    #     if args.ZINC_filtered == True:
    #         with open(cwd + '/data/NEW_ZINC_FILTERED', 'rb') as fp:
    #             zinc_mols = pickle.load(fp)

    #     dataset = load_zinc(zinc_mols)  # pyg dataloader
    #     dataset_smiles = get_smiles(dataset)
    #     shuffle(dataset)

    #     dataset = dataset[:1000]

    #     if args.test_set == True:
    #         with open(cwd + '/data/ZINC_mols_test', 'rb') as fp:
    #             zinc_mols_test = pickle.load(fp)
    #             dataset_test = load_zinc(zinc_mols_test)
    #     else:
    #         dataset_test = False

    guacm_smiles = "/home/nobilm@usi.ch/master_thesis/guacamol/testdata.smiles"
    guac_mols = mols_from_file(guacm_smiles, True)
    atom2num, num2atom, max_num_node = get_atoms_info(guac_mols)
    bond2num = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    num2bond = {v: k for k, v in bond2num.items()}

    data = rdkit2pyg([guac_mols[0]]) # HERE

    if args.supervised:
        supervised_training(data, device, cuda, train_log, test_log, [], [])



