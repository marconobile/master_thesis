import torch
from torch_geometric.nn.conv import NNConv
from torch import nn
import torch.nn.functional as F
import torch_scatter


# maybe conditionals attributes for gan vs wgan
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        nn1 = nn.Sequential(nn.Linear(args.edge_feature_dims-1, 128), nn.LeakyReLU(), nn.Linear(128, args.node_feature_dims * 128))
        self.conv1 = NNConv(args.node_feature_dims, 128, nn1, aggr='mean', root_weight=True)

        nn2 = nn.Sequential(nn.Linear(args.edge_feature_dims-1, 256), nn.LeakyReLU(), nn.Linear(256, (128 + args.node_feature_dims) * 128))
        self.conv2 = NNConv(128 + args.node_feature_dims, 128, nn2, aggr='mean', root_weight=True)

        nn3 = nn.Sequential(nn.Linear(args.edge_feature_dims-1, 256), nn.LeakyReLU(), nn.Linear(256, (128 + args.node_feature_dims) * 256))
        self.conv3 = NNConv(128 + args.node_feature_dims, 256, nn3, aggr='mean', root_weight=True)

        self.fc1 = torch.nn.Linear(256 + args.node_feature_dims, 256)
        self.fc2 = torch.nn.Linear(256, 512)
        self.fc3 = torch.nn.Linear(512, 1)

        self.args = args
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        data1 = F.leaky_relu(self.conv1(data.x.to(torch.float32), data.edge_index, data.edge_attr))

        data1 = torch.cat([data1, data.x.to(torch.float32)], dim=-1)

        data2 = F.leaky_relu(self.conv2(data1, data.edge_index, data.edge_attr))

        data2 = torch.cat([data2, data.x.to(torch.float32)], dim=-1)

        data3 = F.leaky_relu(self.conv3(data2, data.edge_index, data.edge_attr))

        data3 = torch.cat([data3, data.x.to(torch.float32)], dim=-1)

        x = torch_scatter.scatter_mean(data3, index=data.batch, dim=0)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)


# class Reward_Net_Single(nn.Module):
#     def __init__(self, args):
#         super(Reward_Net_Single, self).__init__()
#         nn1 = nn.Sequential(nn.Linear(4, 128), nn.LeakyReLU(), nn.Linear(128, args.node_feature_dims * 128))
#         self.conv1 = NNConv(args.node_feature_dims, 128, nn1, aggr='mean', root_weight=True)
#
#         nn2 = nn.Sequential(nn.Linear(4, 256), nn.LeakyReLU(), nn.Linear(256, (128 + args.node_feature_dims) * 128))
#         self.conv2 = NNConv(128 + args.node_feature_dims, 128, nn2, aggr='mean', root_weight=True)
#
#         nn3 = nn.Sequential(nn.Linear(4, 256), nn.LeakyReLU(), nn.Linear(256, (128 + args.node_feature_dims) * 256))
#         self.conv3 = NNConv(128 + args.node_feature_dims, 256, nn3, aggr='mean', root_weight=True)
#
#         self.fc1 = torch.nn.Linear(256 + args.node_feature_dims, 256)
#         self.fc2 = torch.nn.Linear(256, 512)
#         self.fc3 = torch.nn.Linear(512, 1)
#         self.args = args
#         self.dropout = torch.nn.Dropout(p=0.5)
#
#     def forward(self, data):
#         data1 = F.leaky_relu(self.conv1(data.x.to(torch.float32), data.edge_index, data.edge_attr))
#
#         data1 = torch.cat([data1, data.x.to(torch.float32)], dim=-1)
#
#         data2 = F.leaky_relu(self.conv2(data1, data.edge_index, data.edge_attr))
#
#         data2 = torch.cat([data2, data.x.to(torch.float32)], dim=-1)
#
#         data3 = F.leaky_relu(self.conv3(data2, data.edge_index, data.edge_attr))
#
#         data3 = torch.cat([data3, data.x.to(torch.float32)], dim=-1)
#
#         x = torch_scatter.scatter_mean(data3, index=data.batch, dim=0)
#
#         x = F.leaky_relu(self.fc1(x))
#         x = F.leaky_relu(self.fc2(x))
#         if self.args.reward_type == 'valid':
#             return torch.sigmoid(self.fc3(x))
#         if self.args.reward_type == 'fg':
#             return self.fc3(x)

# class Reward_Net_Joint(nn.Module):
#     def __init__(self):
#         super(Reward_Net_Joint, self).__init__()
#         nn1 = nn.Sequential(nn.Linear(4, 256), nn.LeakyReLU(), nn.Linear(256, 4 * 256))
#         self.conv1 = NNConv(4, 256, nn1, aggr='mean', root_weight=True)
#
#         nn2 = nn.Sequential(nn.Linear(4, 512), nn.LeakyReLU(), nn.Linear(512, 512 * 256))
#         self.conv2 = NNConv(256, 512, nn2, aggr='mean', root_weight=True)
#
#         self.fc1 = torch.nn.Linear(512, 768)
#         self.fc2 = torch.nn.Linear(768, 1024)
#         self.fc3 = torch.nn.Linear(1024, 3)
#
#     def forward(self, data):
#         data1 = F.leaky_relu(self.conv1(data.x.to(torch.float32), data.edge_index, data.edge_attr))
#         data2 = F.leaky_relu(self.conv2(data1, data.edge_index, data.edge_attr))
#         x = torch_scatter.scatter_mean(data2, index=data.batch, dim=0)
#         x = F.leaky_relu(self.fc1(x))
#         x = F.leaky_relu(self.fc2(x))
#         return F.sigmoid(self.fc3(x))
#
