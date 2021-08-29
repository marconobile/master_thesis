import torch
from torch_sparse import coalesce
from torch_scatter import scatter_add #, scatter_max, scatter_mean
from torch_geometric.utils import remove_self_loops


class LineGraph(object):

    def __init__(self, force_directed=False):
        self.force_directed = force_directed

    def __call__(self, data):
        N = data.num_nodes  # gen num of nodes of the given graph
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr  # creates local 2 vars for the graph attrs
        (row, col), edge_attr = coalesce(edge_index, edge_attr, N, N, op='max')  # rimuove doppi
        # represents the edg atrs in COO format as:
        # sparse.coo_matrix((data, (row, col)), shape=(4, 4))

        # Compute node indices.
        mask = row < col  # upper triang
        row, col = row[mask], col[mask]  # tiene solo i true cme indx
        i = torch.arange(row.size(0), dtype=torch.long, device=row.device)  #

        (row, col), i = coalesce(
            torch.stack([
                torch.cat([row, col], dim=0),  # for simmetry
                torch.cat([col, row], dim=0)
            ], dim=0), torch.cat([i, i], dim=0), N, N)

        # Compute new edge indices according to `i`.
        count = scatter_add(torch.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes)
        # conta quante volte un nodo appare cme indice i # num di outgoing edges from a node
        joints = torch.split(i, count.tolist())

        def generate_grid(j, x):
            row = j.view(-1, 1).repeat(1, j.numel()).view(-1)
            col = j.repeat(j.numel())
            edge_attr = x.view(1, -1).repeat(col.numel(), 1)
            return torch.stack([row, col], dim=0), edge_attr

        joints, joints_edge_attr = list(zip(*[generate_grid(joint, xx) for joint, xx in zip(joints, x)]))
        joints = torch.cat(joints, dim=1)
        joints_edge_attr = torch.cat(joints_edge_attr, dim=0)
        joints, joints_edge_attr = remove_self_loops(joints, joints_edge_attr)
        N = row.size(0) // 2
        joints, joints_edge_attr = coalesce(joints, joints_edge_attr, N, N, op='max')

        if edge_attr is not None:
            data.x = scatter_add(edge_attr, i, dim=0, dim_size=N)
            # data.x, _ = scatter_max(edge_attr, i, dim=0, dim_size=N)

        data.edge_index = joints
        data.num_nodes = edge_index.size(1) // 2

        data.edge_attr = joints_edge_attr
        data.edge_map = i
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
