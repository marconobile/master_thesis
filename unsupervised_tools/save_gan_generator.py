import os
import torch

def save_model(ECC_nodes, ECC_edges, epoch):
    path = os.getcwd() + "/weights/"
    checkpoint_ecc_nodes = {
        'model_state_dict': ECC_nodes.state_dict(),
    }
    torch.save(checkpoint_ecc_nodes, path + f'ECC_nodes_checkpoint_{epoch}.pth')

    checkpoint_ecc_edges = {
        'model_state_dict': ECC_edges.state_dict(),
    }

    torch.save(checkpoint_ecc_edges, path + f'ECC_edges_checkpoint_{epoch}.pth')
    print(f'Model saved at epoch {epoch}')
    
    return None
