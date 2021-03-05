import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_graph(adj, nf, ef, name):
    NUM_TO_SYMBOL = {0: 'C', 1: 'N', 2: 'O', 3: 'F'}
    dict_nodes = {}

    for ni, row in enumerate(nf):
        dict_nodes[ni] = NUM_TO_SYMBOL[row.item()]

    BOND_MAP = {  # 0: rdc.rdchem.BondType.ZERO,
        0: "S",
        1: "D",
        2: "T",
        3: "A"}
    from_, to_ = np.nonzero(adj)
    edges = []
    edge_labels = {}
    for idx in range(np.nonzero(adj)[0].shape[0]):
        edges.append([str(from_[idx]), str(to_[idx])])
        edge_labels[(str(from_[idx]), str(to_[idx]))] = BOND_MAP[ef[from_[idx], to_[idx]][0]]

    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1, \
            node_size=500, node_color='pink', alpha=0.9, \
            labels={node: dict_nodes[int(node)] for node in G.nodes()})
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.axis('off')
    plt.show()
    plt.savefig(name)
