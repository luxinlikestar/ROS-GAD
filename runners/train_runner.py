import torch
import numpy as np
from utils.data import ad_split_num
from models.simgrace import simclr, gen_ran_output
from models.GraphCL import GraphCL, train_graphcl_node_level, prepare_augmented_data, normalize_adj, sparse_mx_to_torch_sparse_tensor
import dgl
from utils.utils import mrq_pool_1hop_embeddings_batch, softmax_weighted_1hop_agg_batch, PromptGenerator
from torch_geometric.utils import to_scipy_sparse_matrix
class TrainRunner:
    def __init__(self, graph, labels, dset_info, anomaly_info, args):
        self.x_all = graph.x.float()
        self.edge_index = graph.edge_index
        self.device = args.device
        if torch.is_tensor(labels):
            self.labels = labels.clone().detach().flatten().squeeze()
        else:
            self.labels = torch.tensor(labels.flatten()).squeeze()
        self.dset_info = dset_info
        self.anomaly_info = anomaly_info
        self.args = args
        self.split_info = ad_split_num(self.labels, args.train_ratio, args.num_anomaly, self.anomaly_info)
        print("num known anomalies: %d" % (len(self.split_info['idx_anomaly_train'])))
        def binarize_labels(labels, anomaly_indices, normal_indices):
            if isinstance(labels, np.ndarray):
                labels = torch.tensor(labels, dtype=torch.long)
            binary_labels = torch.zeros_like(labels, dtype=torch.long)
            binary_labels[anomaly_indices] = 1
            binary_labels[normal_indices] = 0
            return binary_labels
        self.labels = binarize_labels(self.labels, np.hstack(
            (self.split_info['idx_anomaly_train'], self.split_info['idx_test']['known_only'],
             self.split_info['idx_test']['unknown_only'])),
                                      np.hstack(
                                          (self.split_info['idx_normal_train'], self.split_info['idx_test']['normal'],
                                           self.split_info['idx_val1'], self.split_info['idx_val2'])))
    def pretrain_simgrace(self):
        print("Starting SimGRACE model pretraining...")
        train_idx = self.split_info['idx_train']
        print(f"using {len(train_idx)} nodes pretraining")
        from torch_geometric.data import Data
        from torch_geometric.utils import subgraph
        train_edge_index, _ = subgraph(
            train_idx,
            self.edge_index,
            relabel_nodes=True,
            num_nodes=self.x_all.size(0)
        )
        train_x = self.x_all[train_idx]
        train_data = Data(x=train_x, edge_index=train_edge_index)
        train_data = train_data.to(self.device)
        num_features = train_data.x.size(1)
        model = simclr(num_features, self.args.hidden_dim, self.args.n_layers, encoder_type='graphsage').to(self.device)
        vice_model = simclr(num_features, self.args.hidden_dim, self.args.n_layers, encoder_type='graphsage').to(
            self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        model.train()
        for epoch in range(1, self.args.epochs + 1):
            optimizer.zero_grad()
            x2 = gen_ran_output(train_data, model, vice_model, self.args)
            x1 = model(train_data.x, train_data.edge_index)
            loss = model.loss_cal(x2, x1)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
        print("pretraining completed！")
        return model

    def loss(self, alpha=1.0, lambda_weight=0.001):
        normal_embeddings = self.combined_val1
        anomaly_embeddings = self.anomaly_prototypes
        normal_prototype = self.normal_prototype
        if anomaly_embeddings.size(0) == 0:
            print("WARNING: No anomaly prototypes generated!")
        normal_distances = torch.norm(normal_embeddings - normal_prototype.unsqueeze(0), dim=1)
        anomaly_distances = torch.norm(anomaly_embeddings - normal_prototype.unsqueeze(0), dim=1)
        anomaly_loss = torch.mean(torch.clamp(alpha - anomaly_distances, min=0.0))
        anomaly_loss = lambda_weight * anomaly_loss
        normal_loss = torch.mean(normal_distances)
        total_loss = anomaly_loss + normal_loss
        return total_loss, normal_loss, anomaly_loss
    def pretrain_grapgcl(self):
        print("Starting GraphCL model pretraining...")
        train_idx = self.split_info['idx_train']
        print(f"using {len(train_idx)} nodes pretraining...")
        from torch_geometric.data import Data
        from torch_geometric.utils import subgraph
        train_edge_index, _ = subgraph(
            train_idx,
            self.edge_index,
            relabel_nodes=True,
            num_nodes=self.x_all.size(0)
        )
        train_x = self.x_all[train_idx]
        train_data = Data(x=train_x, edge_index=train_edge_index)
        train_data = train_data.to(self.device)
        num_features = train_data.x.size(1)
        num_nodes = train_data.x.size(0)
        adj_sparse = to_scipy_sparse_matrix(train_data.edge_index, num_nodes=num_nodes)
        adj_normalized = normalize_adj(adj_sparse)
        adj_torch = sparse_mx_to_torch_sparse_tensor(adj_normalized).to(self.device)
        features = train_data.x.unsqueeze(0)
        model = GraphCL(n_in=num_features, n_h=self.args.hidden_dim, activation='prelu').to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()
        model.train()
        for epoch in range(1, self.args.epochs + 1):
            aug_features1, aug_features2, aug_adj1, aug_adj2 = prepare_augmented_data(
                features.cpu().numpy() if features.is_cuda else features.numpy(),
                adj_sparse,
                aug_type='node',
                drop_percent=0.1
            )
            aug_adj1_normalized = normalize_adj(aug_adj1)
            aug_adj2_normalized = normalize_adj(aug_adj2)
            aug_adj1_torch = sparse_mx_to_torch_sparse_tensor(aug_adj1_normalized).to(self.device)
            aug_adj2_torch = sparse_mx_to_torch_sparse_tensor(aug_adj2_normalized).to(self.device)
            if isinstance(aug_features1, np.ndarray):
                aug_features1 = torch.from_numpy(aug_features1).float().to(self.device)
                aug_features2 = torch.from_numpy(aug_features2).float().to(self.device)
            else:
                aug_features1 = aug_features1.to(self.device)
                aug_features2 = aug_features2.to(self.device)
            loss = train_graphcl_node_level(
                model=model,
                features=features,
                adj=adj_torch,
                aug_features1=aug_features1,
                aug_features2=aug_features2,
                aug_adj1=aug_adj1_torch,
                aug_adj2=aug_adj2_torch,
                sparse=True,
                optimizer=optimizer,
                criterion=criterion,
                nb_nodes=num_nodes,
                batch_size=1,
                aug_type='node',
                device=self.device
            )
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss:.4f}')
        print("pretraining completed！")
        return model
    def train(self):
        if self.args.pretrain == 'simgrace':
            pretrained_model = self.pretrain_simgrace().to(self.device)
        elif self.args.pretrain == 'graphcl':
            pretrained_model = self.pretrain_grapgcl().to(self.device)
        pretrained_model.eval()
        for param in pretrained_model.parameters():
            param.requires_grad = False
        from torch_geometric.data import Data
        pyg_graph = Data(x=self.x_all, edge_index=self.edge_index).to(self.device)
        with torch.no_grad():
            if self.args.pretrain == 'simgrace':
                node_embeds = pretrained_model(pyg_graph.x, pyg_graph.edge_index)
            elif self.args.pretrain == 'graphcl':
                num_nodes = pyg_graph.x.size(0)
                adj_sparse = to_scipy_sparse_matrix(pyg_graph.edge_index, num_nodes=num_nodes)
                adj_normalized = normalize_adj(adj_sparse)
                adj_torch = sparse_mx_to_torch_sparse_tensor(adj_normalized).to(self.device)
                features = pyg_graph.x.unsqueeze(0)
                h, c = pretrained_model.embed(features, adj_torch, sparse=True)
                node_embeds = h.squeeze(0)
            else:
                node_embeds = pretrained_model(pyg_graph.x, pyg_graph.edge_index)
        self.base_node_embeds = node_embeds
        src, dst = self.edge_index
        g_dgl = dgl.graph((src, dst), num_nodes=self.x_all.size(0)).to(self.device)
        g_dgl.ndata['embeds'] = node_embeds
        idx_train = self.split_info['idx_train']
        idx_val1 = self.split_info['idx_val1']
        idx_val2 = self.split_info['idx_val2']
        test_all_idx = self.split_info['idx_test']['all']
        test_unknown_idx = self.split_info['idx_test']['unknown']
        embed_dim = node_embeds.size(1)
        prompt_dim = embed_dim
        self.prompt_gen = PromptGenerator(prompt_dim, num_prompts=self.args.num_prompts).to(self.device)
        self.combine_mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * embed_dim, 2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim)
        ).to(self.device)
        self.all_ids = torch.arange(self.x_all.size(0), device=self.device)
        print("Starting HeAggregator...")
        self.pooled_all = mrq_pool_1hop_embeddings_batch(g_dgl, self.all_ids, embed_key='embeds')
        print("Starting HoAggregator...")
        self.weighted_all = softmax_weighted_1hop_agg_batch(g_dgl, self.all_ids, embed_key='embeds')

        base_pooled_weighted = torch.cat([node_embeds, self.pooled_all, self.weighted_all], dim=1)
        reduced_embeds = self.combine_mlp(base_pooled_weighted)
        prompt_all, _ = self.prompt_gen(node_embeds, self.pooled_all, self.weighted_all)
        combined_all = torch.cat([reduced_embeds, prompt_all], dim=1)
        g_dgl.ndata['embeds'] = combined_all
        self.prompt_graph = g_dgl
        self.combined_train = combined_all[idx_train.to(self.device)]
        self.combined_val1 = combined_all[idx_val1.to(self.device)]
        self.combined_val2 = combined_all[idx_val2.to(self.device)]
        self.combined_test_all = combined_all[test_all_idx.to(self.device)]
        if test_unknown_idx.numel() == 0:
            print("-------No unknown anomaly nodes in test set-------")
            self.combined_test_unknown = combined_all.new_zeros((0, combined_all.size(1)))
        else:
            self.combined_test_unknown = combined_all[test_unknown_idx.to(self.device)]
        self.normal_prototype = self.combined_val1.mean(dim=0)
        self.embed_dim = embed_dim
        self.combined_dim = combined_all.size(1)
        def get_anomaly_prototypes():
            num_anomaly_prototypes = len(self.split_info['idx_val1'])
            prompt_all, _ = self.prompt_gen(node_embeds, self.pooled_all, self.weighted_all)
            padded_prompts = torch.zeros(num_anomaly_prototypes, self.combined_dim, device=self.device)
            padded_prompts[:, -self.embed_dim:] = prompt_all[idx_val1.to(self.device)]
            return padded_prompts
        self.get_anomaly_prototypes = get_anomaly_prototypes
        self.anomaly_prototypes = get_anomaly_prototypes()
        metrics = self.train_prompt_vectors(alpha=self.args.alpha, lambda_weight=self.args.lambda_weight,
                                            lr=self.args.prompt_lr, epochs=self.args.prompt_epochs)
        return metrics
    def train_prompt_vectors(self, alpha=1.0, lambda_weight=0.001, lr=0.001, epochs=100, temperature=0.1):
        print("Starting training of learnable prompt vectors...")
        params = list(self.prompt_gen.parameters()) + list(self.combine_mlp.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=5e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        metrics = {'all_auc_roc': 0, 'all_auc_pr': 0, 'unknown_auc_roc': 0, 'unknown_auc_pr': 0, 'normal_FPR_all': 0, 'normal_FPR_unknown': 0,}

        for epoch in range(epochs):
            optimizer.zero_grad()
            self.anomaly_prototypes = self.get_anomaly_prototypes()
            base_pooled_weighted = torch.cat([self.base_node_embeds, self.pooled_all, self.weighted_all], dim=1)
            reduced_embeds = self.combine_mlp(base_pooled_weighted)
            prompt_all, _ = self.prompt_gen(self.base_node_embeds, self.pooled_all, self.weighted_all)
            combined_all = torch.cat([reduced_embeds, prompt_all], dim=1)
            self.prompt_graph.ndata['embeds'] = combined_all
            idx_val1 = self.split_info['idx_val1'].to(self.device)
            idx_val2 = self.split_info['idx_val2'].to(self.device)
            test_all_idx = self.split_info['idx_test']['all'].to(self.device)
            test_unknown_idx = self.split_info['idx_test']['unknown'].to(self.device)
            self.combined_val1 = combined_all[idx_val1].clone()
            self.combined_val2 = combined_all[idx_val2].clone()
            self.combined_test_all = combined_all[test_all_idx].clone()
            self.combined_test_unknown = combined_all[test_unknown_idx].clone()
            self.normal_prototype = self.combined_val1.mean(dim=0)
            total_loss, normal_sim_loss, anomaly_sim_loss = self.loss(alpha, lambda_weight)
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"WARNING: Invalid loss detected at epoch {epoch}: {total_loss.item()}")
                continue
            total_loss.backward()
            total_grad_norm = 0
            has_grad = False
            for name, param in self.prompt_gen.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2)
                    total_grad_norm += grad_norm.item() ** 2
                    has_grad = True
                    if epoch % 20 == 0 or epoch < 5:
                        print(f"  {name}: grad_norm={grad_norm.item():.6f}")
                else:
                    if epoch % 20 == 0 or epoch < 5:
                        print(f"  {name}: NO GRADIENT!")
            total_grad_norm = total_grad_norm ** (1. / 2)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.4f}, Grad Norm: {total_grad_norm:.4f}, Normal Sim Loss: {normal_sim_loss.item():.4f}, Anomaly Sim Loss: {anomaly_sim_loss.item():.4f}")

        self.anomaly_prototypes = self.get_anomaly_prototypes()

        metrics = self.conformal_evaluate()

        return metrics
    def conformal_evaluate(self):
        import numpy as np
        import torch
        epsilon = getattr(self.args, 'epsilon', 0.1)
        def anomaly_score(embeddings):
            euclidean_dist = torch.norm(embeddings - self.normal_prototype.unsqueeze(0), dim=1)
            return euclidean_dist
        calib_scores = anomaly_score(self.combined_val2).detach().cpu().numpy()
        test_all_scores = anomaly_score(self.combined_test_all).detach().cpu().numpy()
        test_unknown_scores = anomaly_score(self.combined_test_unknown).detach().cpu().numpy()
        def p_values(scores, calib):
            calib_sorted = np.sort(calib)
            n = calib_sorted.shape[0]
            idx = np.searchsorted(calib_sorted, scores, side="left")
            ge_count = n - idx
            pvals = (ge_count + 1) / (n + 1)
            return pvals
        p_all = p_values(test_all_scores, calib_scores)
        p_unknown = p_values(test_unknown_scores, calib_scores)
        y_pred_all = (p_all < epsilon).astype(int)
        y_pred_unknown = (p_unknown < epsilon).astype(int)
        idx_all = self.split_info['idx_test'].get('all')
        idx_unknown = self.split_info['idx_test'].get('unknown')
        y_true_all = self.labels[idx_all].detach().cpu().numpy()
        y_true_unknown = self.labels[idx_unknown].detach().cpu().numpy()
        metrics = {}
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            metrics['all_auc_roc'] = roc_auc_score(y_true_all, y_pred_all)
            metrics['all_auc_pr'] = average_precision_score(y_true_all, y_pred_all)
        except Exception:
            metrics['all_auc_roc'] = float('nan')
            metrics['all_auc_pr'] = float('nan')
        try:
            metrics['unknown_auc_roc'] = roc_auc_score(y_true_unknown, y_pred_unknown)
            metrics['unknown_auc_pr'] = average_precision_score(y_true_unknown, y_pred_unknown)
        except Exception:
            metrics['unknown_auc_roc'] = float('nan')
            metrics['unknown_auc_pr'] = float('nan')
        try:
            normal_mask_all = (y_true_all == 0)
            normal_mask_unknown = (y_true_unknown == 0)
            if normal_mask_all.sum() > 0:
                y_pred_normals_all = y_pred_all[normal_mask_all]
                fpr_all = np.mean(y_pred_normals_all == 1)
            else:
                fpr_all = float('nan')
            if normal_mask_unknown.sum() > 0:
                y_pred_normals_unknown = y_pred_unknown[normal_mask_unknown]
                fpr_unknown = np.mean(y_pred_normals_unknown == 1)
            else:
                fpr_unknown = float('nan')
            metrics['normal_FPR_all'] = fpr_all
            metrics['normal_FPR_unknown'] = fpr_unknown
        except Exception as e:
            print("Warning: FPR computation failed:", e)
            metrics['normal_FPR_all'] = float('nan')
            metrics['normal_FPR_unknown'] = float('nan')
        print(f"Conformal epsilon={epsilon}")
        print(metrics)
        return metrics