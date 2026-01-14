import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import copy
import random
def aug_random_mask(input_feature, drop_percent=0.2):
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature
def aug_random_edge(input_adj, drop_percent=0.2):
    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()
    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))
    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))
    edge_num = int(len(row_idx) / 2)
    add_drop_num = int(edge_num * percent / 2)
    aug_adj = copy.deepcopy(input_adj.todense().tolist())
    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)
    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)
    for i in add_list:
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1
    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj
def aug_drop_node(input_fea, input_adj, drop_percent=0.2):
    input_adj = torch.tensor(input_adj.todense().tolist())
    if isinstance(input_fea, np.ndarray):
        input_fea = torch.from_numpy(input_fea).float()
    if len(input_fea.shape) == 3:
        input_fea = input_fea.squeeze(0)
    elif len(input_fea.shape) == 2:
        pass
    else:
        raise ValueError(f"Unexpected input_fea shape: {input_fea.shape}")
    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)
    all_node_list = [i for i in range(node_num)]
    drop_node_list = sorted(random.sample(all_node_list, drop_num))
    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)
    if isinstance(aug_input_fea, torch.Tensor):
        aug_input_fea = aug_input_fea.numpy()
    aug_input_fea = np.expand_dims(aug_input_fea, axis=0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))
    return aug_input_fea, aug_input_adj
def aug_subgraph(input_fea, input_adj, drop_percent=0.2):
    input_adj = torch.tensor(input_adj.todense().tolist())
    if isinstance(input_fea, np.ndarray):
        input_fea = torch.from_numpy(input_fea).float()
    if len(input_fea.shape) == 3:
        input_fea = input_fea.squeeze(0)
    elif len(input_fea.shape) == 2:
        pass
    else:
        raise ValueError(f"Unexpected input_fea shape: {input_fea.shape}")
    node_num = input_fea.shape[0]
    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * (1 - drop_percent))
    center_node_id = random.randint(0, node_num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []
    for i in range(s_node_num - 1):
        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()
        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break
    drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])
    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)
    if isinstance(aug_input_fea, torch.Tensor):
        aug_input_fea = aug_input_fea.numpy()
    aug_input_fea = np.expand_dims(aug_input_fea, axis=0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))
    return aug_input_fea, aug_input_adj
def delete_row_col(input_matrix, drop_list, only_row=False):
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]
    return out
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)
        return logits
class Discriminator2(nn.Module):
    def __init__(self, n_h):
        super(Discriminator2, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)
        return logits
class GraphCL(nn.Module):
    def __init__(self, n_in, n_h, activation='prelu'):
        super(GraphCL, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.disc2 = Discriminator2(n_h)
    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse,
                msk=None, samp_bias1=None, samp_bias2=None, aug_type='edge'):
        h_0 = self.gcn(seq1, adj, sparse)
        if aug_type == 'edge':
            h_1 = self.gcn(seq1, aug_adj1, sparse)
            h_3 = self.gcn(seq1, aug_adj2, sparse)
        elif aug_type == 'mask':
            h_1 = self.gcn(seq3, adj, sparse)
            h_3 = self.gcn(seq4, adj, sparse)
        elif aug_type == 'node' or aug_type == 'subgraph':
            h_1 = self.gcn(seq3, aug_adj1, sparse)
            h_3 = self.gcn(seq4, aug_adj2, sparse)
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)
        c_3 = self.read(h_3, msk)
        c_3 = self.sigm(c_3)
        h_2 = self.gcn(seq2, adj, sparse)
        ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)
        ret = ret1 + ret2
        return ret
    def embed(self, seq, adj, sparse, msk=None):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)
        return h_1.detach(), c.detach()
def train_graphcl_node_level(model, features, adj, aug_features1, aug_features2,
                             aug_adj1, aug_adj2, sparse, optimizer, criterion,
                             nb_nodes, batch_size=1, aug_type='edge', device='cuda'):
    model.train()
    optimizer.zero_grad()
    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]
    if isinstance(device, torch.device):
        target_device = device
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = features.device
    lbl_1 = torch.ones(batch_size, nb_nodes, device=target_device)
    lbl_2 = torch.zeros(batch_size, nb_nodes, device=target_device)
    lbl = torch.cat((lbl_1, lbl_2), 1)
    shuf_fts = shuf_fts.to(target_device)
    logits = model(features, shuf_fts, aug_features1, aug_features2,
                   adj if not sparse else adj,
                   aug_adj1 if not sparse else aug_adj1,
                   aug_adj2 if not sparse else aug_adj2,
                   sparse, None, None, None, aug_type=aug_type)
    loss = criterion(logits, lbl)
    loss.backward()
    optimizer.step()
    return loss.item()
def prepare_augmented_data(features, adj, aug_type='edge', drop_percent=0.1):
    if aug_type == 'edge':
        aug_features1 = features
        aug_features2 = features
        aug_adj1 = aug_random_edge(adj, drop_percent=drop_percent)
        aug_adj2 = aug_random_edge(adj, drop_percent=drop_percent)
    elif aug_type == 'node':
        aug_features1, aug_adj1 = aug_drop_node(features, adj, drop_percent=drop_percent)
        aug_features2, aug_adj2 = aug_drop_node(features, adj, drop_percent=drop_percent)
    elif aug_type == 'subgraph':
        aug_features1, aug_adj1 = aug_subgraph(features, adj, drop_percent=drop_percent)
        aug_features2, aug_adj2 = aug_subgraph(features, adj, drop_percent=drop_percent)
    elif aug_type == 'mask':
        aug_features1 = aug_random_mask(features, drop_percent=drop_percent)
        aug_features2 = aug_random_mask(features, drop_percent=drop_percent)
        aug_adj1 = adj
        aug_adj2 = adj
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
    return aug_features1, aug_features2, aug_adj1, aug_adj2
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    rowsum = rowsum.flatten()
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
    nonzero_mask = rowsum > 0
    d_inv_sqrt[nonzero_mask] = np.power(rowsum[nonzero_mask], -0.5)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)