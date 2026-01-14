import argparse
import random

from utils.io import load_dgl_graph, dgl_data_to_pyg_graph
import torch
import numpy as np
from runners.train_runner import TrainRunner
from datetime import datetime
import time

def do_exp(graph, labels, dset_info, tau_lower=0.00, tau_upper=0.05, args=None):
    print("Treat any class whose proportion %f <= %f as anomaly classes" % (tau_lower, tau_upper))
    anomaly_class_idx = np.where((dset_info['class_per'] <= tau_upper) & (dset_info['class_per'] >= tau_lower))[0]
    print("Total anomaly percentage %.2f%%" % (dset_info['class_per'][anomaly_class_idx].sum() * 100))

    print("anomaly classes: %s" % str(anomaly_class_idx.tolist()))

    all_results = []
    for idx in anomaly_class_idx:
        class_name = dset_info['class_names'][idx] if dset_info['class_names'] is not None else str(idx)
        print("using class %d: %s as known anomalies" % (idx, class_name))
        anomaly_info = {'known_anomaly': idx,
                        'unknown_anomaly': [i for i in anomaly_class_idx if i != idx],
                        'normal': [i for i in dset_info['class_idx'].tolist() if i not in anomaly_class_idx],
                        'all_anomaly': anomaly_class_idx}
        runner = TrainRunner(graph, labels, dset_info, anomaly_info, args)
        metrics = runner.train()

        metrics['anomaly_class'] = idx
        metrics['class_name'] = class_name
        all_results.append(metrics)
        print(f"anomaly class {idx} ({class_name}) training completed")

    avg_test_all_roc = np.mean([result['all_auc_roc'] for result in all_results])
    avg_test_all_pr = np.mean([result['all_auc_pr'] for result in all_results])
    avg_test_unknown_roc = np.mean([result['unknown_auc_roc'] for result in all_results])
    avg_test_unknown_pr = np.mean([result['unknown_auc_pr'] for result in all_results])
    avg_normal_fpr_all = np.mean([result['normal_FPR_all'] for result in all_results])
    avg_normal_fpr_unknown = np.mean([result['normal_FPR_unknown'] for result in all_results])

    return {'auroc_test_all': avg_test_all_roc, 'aupr_test_all': avg_test_all_pr, 'aupr_test_un': avg_test_unknown_pr, 'auroc_test_un': avg_test_unknown_roc, 'normal_fpr_all': avg_normal_fpr_all, 'normal_fpr_unknown': avg_normal_fpr_unknown}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument('--dset_name', type=str, default='cs', help='dataset name')
    parser.add_argument('--epochs', type=int, default=200, help='Training epoch')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--drop_out', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--train_ratio', default='0.05', type=float, help='train_ratio')
    parser.add_argument('--num_anomaly', default='50', type=int, help='num_anomaly')
    parser.add_argument('--hidden_dim', type=int, default=32, help='dimension of hidden embedding')
    parser.add_argument('--n_layers', type=int, default=3, help='number of GNN layers')
    parser.add_argument('--view', type=str, default='net_rur', help='the view of dataset')
    parser.add_argument('--eta', type=float, default=1.0, help='Gaussian perturbation scale for vice model')
    parser.add_argument('--pretrain', type = str, default='simgrace', help='pretrain model')
    parser.add_argument('--alpha', type=float, default=2.0, help='margin for prototype separation')
    parser.add_argument('--lambda_weight', type=float, default=1.0, help='weight for anomaly term in loss')
    parser.add_argument('--prompt_lr', type=float, default=0.001, help='learning rate for prompt training')
    parser.add_argument('--prompt_epochs', type=int, default=200, help='epochs for prompt training')
    parser.add_argument("--num_prompts", type=int, default=10, help='number of prompts to train')
    parser.add_argument('--epsilon', type=float, default=0.3, help='conformal p-value threshold (0,1)')

    args = parser.parse_args()

    print("Using train.py")

    print(args)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    print("timestamp: %s" % timestamp)
    args.ts = timestamp
    setup_seed(args.seed)

    x_all, adj, class_labels = load_dgl_graph(args.dset_name, homo=1, view=args.view)
    graph,_, dset_info = dgl_data_to_pyg_graph(x_all, adj, class_labels)
    print(dset_info)

    t_start = time.time()
    print("time start:", t_start)

    print(do_exp(graph, class_labels, dset_info, tau_lower=0.0, tau_upper=0.05, args=args))

    t_end = time.time()
    print("time end:", t_end)
    print("Total time: %.2f" % (t_end - t_start))


if __name__ == "__main__":
    run()
