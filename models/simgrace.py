import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, SAGEConv
from torch_geometric.data import Data
LOG_FORMAT = "%(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename='Accuracy.txt', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
def arg_parse():
    import argparse
    parser = argparse.ArgumentParser(description='Node-level SimGRACE (single-file)')
    parser.add_argument('--DS', dest='DS', default='NCI1')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=0.01)
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=2)
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eta', type=float, default=1.0, help='Gaussian perturbation scale for vice model')
    parser.add_argument('--batch_size', type=int, default=128)
    return parser.parse_args()
device = None
class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i:
                nn_seq = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn_seq = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn_seq)
            bn = torch.nn.BatchNorm1d(dim)
            self.convs.append(conv)
            self.bns.append(bn)
    def forward_nodes(self, x, edge_index):
        if x is None:
            num_nodes = int(edge_index.max().item()) + 1
            x = torch.ones((num_nodes, 1), device=edge_index.device)
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)
        return xs[-1]
class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(GraphSAGEEncoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i == 0:
                conv = SAGEConv(num_features, dim, normalize=True)
            else:
                conv = SAGEConv(dim, dim, normalize=True)
            bn = torch.nn.BatchNorm1d(dim)
            self.convs.append(conv)
            self.bns.append(bn)
    def forward_nodes(self, x, edge_index):
        if x is None:
            num_nodes = int(edge_index.max().item()) + 1
            x = torch.ones((num_nodes, 1), device=edge_index.device)
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)
        return xs[-1]
class simclr(nn.Module):
    def __init__(self, num_features, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1, encoder_type='gin'):
        super(simclr, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.embedding_dim = hidden_dim
        if encoder_type == 'graphsage':
            self.encoder = GraphSAGEEncoder(num_features, hidden_dim, num_gc_layers)
        else:
            self.encoder = Encoder(num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.init_emb()
    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    def forward(self, x, edge_index):
        node_emb = self.encoder.forward_nodes(x, edge_index)
        node_emb = self.proj_head(node_emb)
        return node_emb
    def loss_cal(self, x, x_aug):
        T = 0.2
        num = x.size(0)
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        y_norm = x_aug / (x_aug.norm(dim=1, keepdim=True) + 1e-8)
        logits = torch.mm(x_norm, y_norm.t()) / T
        labels = torch.arange(num, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
def gen_ran_output(data, model, vice_model, args):
    for (adv_name, adv_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            adv_param.data = param.data + args.eta * torch.normal(0, torch.ones_like(param.data) * param.data.std()).to(
                next(model.parameters()).device)
    z2 = vice_model(data.x, data.edge_index)
    return z2
class AverageMeter(object):
    def __init__(self, name=None, fmt='.6f'):
        fmtstr = f'{{val:{fmt}}} ({{avg:{fmt}}})'
        if name is not None:
            fmtstr = name + ' ' + fmtstr
        self.fmtstr = fmtstr
        self.reset()
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
    @property
    def avg(self):
        avg = self.sum / self.count
        if isinstance(avg, torch.Tensor):
            avg = avg.item()
        return avg
    def __str__(self):
        val = self.val
        if isinstance(val, torch.Tensor):
            val = val.item()
        return self.fmtstr.format(val=val, avg=self.avg)
class TwoAugUnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return self.transform(image), self.transform(image)
    def __len__(self):
        return len(self.dataset)
def dgl_to_pyg_data(dgl_graph):
    import dgl
    assert isinstance(dgl_graph, dgl.DGLGraph) or isinstance(dgl_graph, dgl.DGLHeteroGraph)
    if isinstance(dgl_graph, dgl.DGLHeteroGraph):
        ntypes = dgl_graph.ntypes
        etypes = dgl_graph.canonical_etypes
        if len(ntypes) != 1 or len(etypes) != 1:
            raise ValueError('HeteroGraph must be reduced to one type before conversion.')
        g = dgl_graph
        x = g.ndata.get('feature', g.ndata.get('feat', None)).float()
        u, v = g.edges()
    else:
        g = dgl_graph
        x = g.ndata.get('feature', g.ndata.get('feat', None)).float()
        u, v = g.edges()
    edge_index = torch.stack([u.long(), v.long()], dim=0)
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data.num_graphs = 1
    return data
def train_on_dgl_graph(dgl_graph, args):
    global device
    setup_seed(args.seed)
    device = torch.device(args.device)
    data = dgl_to_pyg_data(dgl_graph)
    data = data.to(device)
    num_features = data.x.size(1) if data.x is not None else 1
    model = simclr(num_features, args.hidden_dim, args.num_gc_layers).to(device)
    vice_model = simclr(num_features, args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        x2 = gen_ran_output(data, model, vice_model, args)
        x1 = model(data.x, data.edge_index)
        loss = model.loss_cal(x2, x1)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Loss {loss.item():.6f}')
    return model
def train_on_dgl_dataset(dgl_dataset, args):
    return train_on_dgl_graph(dgl_dataset.graph, args)
__all__ = ['simclr', 'train_on_dgl_graph', 'train_on_dgl_dataset', 'dgl_to_pyg_data']
from utils.data import DglDataset
from models.simgrace import train_on_dgl_dataset
class Args:
    device = 'cuda'
    lr = 0.01
    num_gc_layers = 2
    hidden_dim = 32
    seed = 42
    epochs = 100
    eta = 1.0