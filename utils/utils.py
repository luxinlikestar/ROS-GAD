import dgl
import torch
import torch.nn as nn
import numpy as np


EPS = 1e-12
def mask_edge(graph, mask_prob):
    E = graph.num_edges()
    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx
def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph
    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]
    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()
    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]
    if return_edges:
        return ng, (dsrc, ddst)
    return ng

def select_mrq_khop(star_khop_graph_big, central_node_id, khop, select_topk):
    device = star_khop_graph_big.device if hasattr(star_khop_graph_big, 'device') else None
    pres = star_khop_graph_big.predecessors(central_node_id)
    sucs = star_khop_graph_big.successors(central_node_id)
    node_ids = torch.unique(torch.cat([pres, sucs], dim=0))
    if node_ids.shape[0] == 0:
        base = torch.tensor([central_node_id]).long()
        w = torch.tensor([[1.0]])
        if device is not None:
            base = base.to(device)
            w = w.to(device)
        return base, w
    central = torch.tensor([central_node_id]).long().to(node_ids.device)
    ego_ids = torch.unique(torch.cat([node_ids, central], dim=0))
    id_map = {int(n.item()): i for i, n in enumerate(ego_ids)}
    src, dst = star_khop_graph_big.edges()
    ego_mask = (torch.isin(src, ego_ids) & torch.isin(dst, ego_ids))
    src_ego = src[ego_mask]
    dst_ego = dst[ego_mask]
    if src_ego.numel() == 0:
        num_nbs = node_ids.shape[0]
        if num_nbs == 0:
            base = torch.tensor([central_node_id]).long().to(node_ids.device)
            w = torch.tensor([[1.0]], device=node_ids.device)
            return base, w
        w = torch.ones(num_nbs, 1) / (num_nbs + EPS)
        nbs = torch.cat([node_ids, central], dim=0)
        weights = torch.cat([0.5 * w, torch.tensor([[0.5]], device=node_ids.device)], dim=0)
        return nbs, weights
    src_loc = torch.tensor([id_map[int(s.item())] for s in src_ego], device=node_ids.device)
    dst_loc = torch.tensor([id_map[int(d.item())] for d in dst_ego], device=node_ids.device)
    n_local = ego_ids.shape[0]
    indices = torch.stack([src_loc, dst_loc], dim=0)
    values = torch.ones(indices.shape[1], device=node_ids.device)
    A = torch.sparse_coo_tensor(indices, values, (n_local, n_local), device=node_ids.device)
    AT = torch.sparse_coo_tensor(indices.flip(0), values, (n_local, n_local), device=node_ids.device)
    A = A + AT
    v = torch.rand(n_local, device=node_ids.device)
    v = v / (torch.norm(v) + EPS)
    num_iter = 10
    for _ in range(num_iter):
        v = torch.sparse.mm(A, v.reshape(-1, 1)).reshape(-1)
        norm = torch.norm(v)
        if norm.item() == 0:
            break
        v = v / norm
    self_loc = id_map[int(central_node_id)]
    scores = v.clone().abs()
    scores[self_loc] = 0
    sum_scores = scores.sum()
    if sum_scores.item() > 0:
        nb_weights = 0.5 * (scores / sum_scores).reshape(-1, 1)
    else:
        mask = torch.ones(n_local, device=node_ids.device)
        mask[self_loc] = 0
        num_nbs = int(mask.sum().item())
        if num_nbs == 0:
            base = torch.tensor([central_node_id]).long().to(node_ids.device)
            w = torch.tensor([[1.0]], device=node_ids.device)
            return base, w
        nb_weights = 0.5 * torch.ones(n_local, 1) / (num_nbs + EPS)
        nb_weights[self_loc] = 0
    nonzero_idx = (nb_weights.reshape(-1) > 0)
    selected_local = torch.nonzero(nonzero_idx).reshape(-1)
    selected_global = ego_ids[selected_local]
    nbs = torch.cat([selected_global, central], dim=0)
    weights = torch.cat([nb_weights[selected_local], torch.tensor([[0.5]], device=node_ids.device)], dim=0)
    return nbs, weights
def mrq_pool_1hop_embeddings(graph: dgl.DGLGraph, central_node_id: int, embed_key: str = 'embeds'):
    with graph.local_scope():
        if embed_key not in graph.ndata:
            raise KeyError(f"Node data missing '{embed_key}' for MRQ pooling")
        nbs, weights = select_mrq_khop(graph, int(central_node_id), khop=1, select_topk=None)
        if not torch.is_tensor(nbs):
            nbs = torch.tensor(nbs).long()
        mask = (nbs != int(central_node_id))
        nbs_no_root = nbs[mask]
        weights_no_root = weights[mask]
        if nbs_no_root.numel() == 0:
            return torch.zeros(graph.ndata[embed_key].shape[1], device=graph.device)
        embeds = graph.ndata[embed_key][nbs_no_root]
        weights_no_root = weights_no_root / (weights_no_root.sum() + 1e-8)
        if weights_no_root.dim() > 1:
            weights_no_root = weights_no_root.squeeze(-1)
        pooled = (embeds * weights_no_root.unsqueeze(-1)).sum(dim=0)
        return pooled
def mrq_pool_1hop_embeddings_batch(graph: dgl.DGLGraph, node_ids: torch.Tensor, embed_key: str = 'embeds'):
    pooled = [mrq_pool_1hop_embeddings(graph, int(nid), embed_key=embed_key) for nid in node_ids]
    return torch.stack(pooled, dim=0) if len(pooled) > 0 else torch.empty(0, graph.ndata[embed_key].shape[1])
def softmax_weighted_1hop_agg(graph: dgl.DGLGraph, central_node_id: int, embed_key: str = 'embeds'):
    with graph.local_scope():
        if embed_key not in graph.ndata:
            raise KeyError(f"Node data missing '{embed_key}' for weighted aggregation")
        z = graph.ndata[embed_key]
        zi = z[central_node_id]
        pres = graph.predecessors(central_node_id)
        sucs = graph.successors(central_node_id)
        nbs = torch.unique(torch.cat([pres, sucs], dim=0))
        nbs = nbs[nbs != int(central_node_id)]
        if nbs.numel() == 0:
            return torch.zeros_like(zi)
        Z_n = z[nbs]
        zi_norm = torch.nn.functional.normalize(zi, dim=0)
        Z_n_norm = torch.nn.functional.normalize(Z_n, dim=1)
        sim = torch.matmul(Z_n_norm, zi_norm)
        sim_shift = sim + 1.0
        weights = torch.softmax(sim_shift, dim=0)
        agg = (weights.unsqueeze(1) * Z_n).sum(dim=0)
        return agg
def softmax_weighted_1hop_agg_batch(graph: dgl.DGLGraph, node_ids: torch.Tensor, embed_key: str = 'embeds'):
    outs = []
    for nid in node_ids.tolist():
        outs.append(softmax_weighted_1hop_agg(graph, int(nid), embed_key=embed_key))
    return torch.stack(outs, dim=0) if len(outs) > 0 else torch.empty(0, graph.ndata[embed_key].shape[1])
class PromptGenerator(nn.Module):
    def __init__(self, embed_dim: int, num_prompts: int = 10, negative_slope: float = 0.2):
        super().__init__()
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        self.prompts = nn.Parameter(torch.randn(num_prompts, embed_dim) * 0.1)
        self.W = nn.Linear(3 * embed_dim, num_prompts)
        self.act = nn.LeakyReLU(negative_slope)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
    def forward(self, z: torch.Tensor, z_pool: torch.Tensor, z_weighted: torch.Tensor):
        x = torch.cat([z, z_pool, z_weighted], dim=1)
        logits = self.W(x)
        attn = torch.softmax(self.act(logits), dim=1)
        enhanced = attn @ self.prompts
        return enhanced, attn
def generate_prompt_vectors(z: torch.Tensor, z_pool: torch.Tensor, z_weighted: torch.Tensor,
                            num_prompts: int = 10, negative_slope: float = 0.2):
    assert z.shape == z_pool.shape == z_weighted.shape
    module = PromptGenerator(z.shape[1], num_prompts=num_prompts, negative_slope=negative_slope).to(z.device)
    enhanced, attn = module(z, z_pool, z_weighted)
    return enhanced, attn, module

class Dataset:
    def split(self, trial_id=0):
        if not self.is_single_graph:
            self.train_labels_dict_list = []
            self.train_graphs = []
            self.train_sp_matrix_graphs = []
            for x in self.graph_train_masks[:, trial_id].nonzero().reshape(-1):
                self.train_graphs.append(self.graph_list[x])
                train_labels_dict = {
                    'node_labels': None,
                    'edge_labels': None,
                    'graph_labels': None,
                }
                if 'n' in self.labels_have:
                    train_labels_dict['node_labels'] = self.graph_list[x].ndata['node_label']
                if 'e' in self.labels_have:
                    train_labels_dict['edge_labels'] = self.graph_list[x].edata['edge_label']
                if 'g' in self.labels_have:
                    train_labels_dict['graph_labels'] = self.graph_label[x]
                self.train_labels_dict_list.append(train_labels_dict)
                self.train_sp_matrix_graphs.append(self.sp_matrix_graph_list[x])
            self.val_labels_dict_list = []
            self.val_graphs = []
            self.val_sp_matrix_graphs = []
            for x in self.graph_val_masks[:, trial_id].nonzero().reshape(-1):
                self.val_graphs.append(self.graph_list[x])
                val_labels_dict = {
                    'node_labels': None,
                    'edge_labels': None,
                    'graph_labels': None,
                }
                if 'n' in self.labels_have:
                    val_labels_dict['node_labels'] = self.graph_list[x].ndata['node_label']
                if 'e' in self.labels_have:
                    val_labels_dict['edge_labels'] = self.graph_list[x].edata['edge_label']
                if 'g' in self.labels_have:
                    val_labels_dict['graph_labels'] = self.graph_label[x]
                self.val_labels_dict_list.append(val_labels_dict)
                self.val_sp_matrix_graphs.append(self.sp_matrix_graph_list[x])
            self.test_labels_dict_list = []
            self.test_graphs = []
            self.test_sp_matrix_graphs = []
            for x in self.graph_test_masks[:, trial_id].nonzero().reshape(-1):
                self.test_graphs.append(self.graph_list[x])
                test_labels_dict = {
                    'node_labels': None,
                    'edge_labels': None,
                    'graph_labels': None,
                }
                if 'n' in self.labels_have:
                    test_labels_dict['node_labels'] = self.graph_list[x].ndata['node_label']
                if 'e' in self.labels_have:
                    test_labels_dict['edge_labels'] = self.graph_list[x].edata['edge_label']
                if 'g' in self.labels_have:
                    test_labels_dict['graph_labels'] = self.graph_label[x]
                self.test_labels_dict_list.append(test_labels_dict)
                self.test_sp_matrix_graphs.append(self.sp_matrix_graph_list[x])
        else:
            self.train_mask_node_cur = self.train_mask_node[trial_id]
            self.train_mask_edge_cur = self.train_mask_edge[trial_id]
            self.train_labels_dict_list = []
            self.train_graphs = []
            self.train_sp_matrix_graphs = []
            for x in self.graph_train_masks[:, trial_id].nonzero().reshape(-1):
                self.train_graphs.append(self.graph_list[x])
                train_labels_dict = {
                    'node_labels': None,
                    'edge_labels': None,
                    'graph_labels': None,
                }
                if 'n' in self.labels_have:
                    train_labels_dict['node_labels'] = self.graph_list[x].ndata['node_label'][self.train_mask_node_cur]
                if 'e' in self.labels_have:
                    train_labels_dict['edge_labels'] = self.graph_list[x].edata['edge_label'][self.train_mask_edge_cur]
                if 'g' in self.labels_have:
                    raise NotImplementedError
                    train_labels_dict['graph_labels'] = self.graph_label[x]
                self.train_labels_dict_list.append(train_labels_dict)
                self.train_sp_matrix_graphs.append(self.sp_matrix_graph_list[0])
            self.val_mask_node_cur = self.val_mask_node[trial_id]
            self.val_mask_edge_cur = self.val_mask_edge[trial_id]
            self.val_labels_dict_list = []
            self.val_graphs = []
            self.val_sp_matrix_graphs = []
            for x in self.graph_val_masks[:, trial_id].nonzero().reshape(-1):
                self.val_graphs.append(self.graph_list[x])
                val_labels_dict = {
                    'node_labels': None,
                    'edge_labels': None,
                    'graph_labels': None,
                }
                if 'n' in self.labels_have:
                    val_labels_dict['node_labels'] = self.graph_list[x].ndata['node_label'][self.val_mask_node_cur]
                if 'e' in self.labels_have:
                    val_labels_dict['edge_labels'] = self.graph_list[x].edata['edge_label'][self.val_mask_edge_cur]
                if 'g' in self.labels_have:
                    raise NotImplementedError
                    val_labels_dict['graph_labels'] = self.graph_label[x]
                self.val_labels_dict_list.append(val_labels_dict)
                self.val_sp_matrix_graphs.append(self.sp_matrix_graph_list[0])
            self.test_mask_node_cur = self.test_mask_node[trial_id]
            self.test_mask_edge_cur = self.test_mask_edge[trial_id]
            self.test_labels_dict_list = []
            self.test_graphs = []
            self.test_sp_matrix_graphs = []
            for x in self.graph_test_masks[:, trial_id].nonzero().reshape(-1):
                self.test_graphs.append(self.graph_list[x])
                test_labels_dict = {
                    'node_labels': None,
                    'edge_labels': None,
                    'graph_labels': None,
                }
                if 'n' in self.labels_have:
                    test_labels_dict['node_labels'] = self.graph_list[x].ndata['node_label'][self.test_mask_node_cur]
                if 'e' in self.labels_have:
                    test_labels_dict['edge_labels'] = self.graph_list[x].edata['edge_label'][self.test_mask_edge_cur]
                if 'g' in self.labels_have:
                    raise NotImplementedError
                    test_labels_dict['graph_labels'] = self.graph_label[x]
                self.test_labels_dict_list.append(test_labels_dict)
                self.test_sp_matrix_graphs.append(self.sp_matrix_graph_list[0])
