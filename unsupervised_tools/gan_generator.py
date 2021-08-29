import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import NNConv
import torch

class Graph_ECC(nn.Module):
    def __init__(self, node_dims, edge_dims):
        super(Graph_ECC, self).__init__()
        nn1 = nn.Sequential(nn.Linear(edge_dims, 512), nn.LeakyReLU(), nn.Linear(512, node_dims * 512))
        self.conv1 = NNConv(node_dims, 512, nn1, aggr='mean', root_weight=True)

        nn2 = nn.Sequential(nn.Linear(edge_dims, 256), nn.LeakyReLU(), nn.Linear(256, 512 * 256))
        self.conv2 = NNConv(512, 256, nn2, aggr='mean', root_weight=True)

        nn3 = nn.Sequential(nn.Linear(edge_dims, 128), nn.LeakyReLU(), nn.Linear(128, 256 * node_dims))
        self.conv3 = NNConv(256, node_dims, nn3, aggr='mean', root_weight=True)

    def forward(self, data, epoch):
        tau = 500/(epoch+1)
        # print("Epoch: ", epoch, " temperature: ", tau)
        data1 = F.leaky_relu(self.conv1(data.x.to(torch.float32), data.edge_index, data.edge_attr))
        data2 = F.leaky_relu(self.conv2(data1, data.edge_index, data.edge_attr))
        data3 = F.leaky_relu(self.conv3(data2, data.edge_index, data.edge_attr))
        data4 = F.gumbel_softmax(data3, dim = -1, hard=True, tau=tau)
        return data4
