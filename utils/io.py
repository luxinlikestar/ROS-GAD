import numpy as np
import torch
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.data import Data
def load_labels(name):
    data = torch.load('../dataset/refined_{}_labels.pt'.format(name))
    labels = data['label']
    anomaly_dist = data['anomaly_dist']
    return labels, anomaly_dist

def load_dgl_graph(dset_name, homo=1, view=None):
    from utils.data import DglDataset
    graph = DglDataset(dset_name, homo=homo, view=view).graph
    x_all, adj = graph.ndata['feature'], graph.adj(scipy_fmt='coo')
    return x_all, adj, graph.ndata['label']
def make_pyg_graph_dgl(x, adj, undirected=True):
    edge_array = np.array([adj.row, adj.col], dtype=np.int64)
    edge_index = torch.from_numpy(edge_array).long()
    if undirected:
        if not is_undirected(edge_index):
            edge_index = to_undirected(edge_index)
    data = Data(x=x, edge_index=edge_index)
    if undirected:
        assert data.is_undirected()
    return data

def dgl_data_to_pyg_graph(x_all, adj, class_labels):
    pyg_graph = make_pyg_graph_dgl(x_all, adj, undirected=True)
    class_idx, class_size = torch.unique(class_labels, return_counts=True)
    class_per = class_size / class_labels.size(0)
    class_names = [f"class_{int(i)}" for i in class_idx.tolist()]
    print(class_per)
    dset_info = {
        'class_idx': class_idx,
        'class_size': class_size,
        'class_per': class_per,
        'class_names': class_names
    }
    return pyg_graph, class_labels, dset_info