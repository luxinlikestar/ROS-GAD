import numpy as np
import torch
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, CoauthorCSDataset
import dgl
from dgl.data.utils import load_graphs
from utils.io import load_labels
def random_split(idx, train_ratio):
    n_train = int(idx.shape[0] * train_ratio)
    randperm = torch.randperm(idx.shape[0])
    return idx[randperm[:n_train]], idx[randperm[n_train:]]
def num_split(idx, n_train):
    randperm = torch.randperm(idx.shape[0])
    return idx[randperm[:n_train]], idx[randperm[n_train:]]
def select_class_idx_in_list(labels, classes):
    node_idx = None
    for i in classes:
        cur_idx = np.where(labels == i)[0].flatten()
        node_idx = cur_idx if node_idx is None else np.hstack((node_idx, cur_idx))
    return node_idx
def get_classes_idx(all_idx, labels, classes):
    selected_idx = None
    for i in classes:
        idx = all_idx[np.where(labels == i)]
        selected_idx = idx if selected_idx is None else np.hstack((selected_idx, idx))
    return selected_idx
def get_split_idx(node_idx, labels, all_classes, known_anomaly_classes, unknown_anomaly_classes):
    normal_classes = [i for i in all_classes if i not in known_anomaly_classes and i not in unknown_anomaly_classes]
    return {
        'normal_idx': get_classes_idx(node_idx, labels, normal_classes),
        'known_idx': get_classes_idx(node_idx, labels, known_anomaly_classes),
        'unknown_idx': get_classes_idx(node_idx, labels, unknown_anomaly_classes),
    }
def ad_split(labels, train_ratio, class_info):
    known_anomaly = class_info['known_anomaly']
    unknown_anomaly_classes = class_info['unknown_anomaly']
    normal_classes = class_info['normal']
    known_anomaly_idx = np.where(labels == known_anomaly)[0].flatten()
    normal_idx = select_class_idx_in_list(labels, normal_classes)
    unknown_anomaly_idx = select_class_idx_in_list(labels, unknown_anomaly_classes)
    normal_train, normal_test = random_split(normal_idx, train_ratio)
    known_anomaly_train, known_anomaly_test = random_split(known_anomaly_idx, train_ratio)
    unknown_anomaly_test = unknown_anomaly_idx
    train_idx = np.hstack((normal_train, known_anomaly_train))
    train_idx = torch.LongTensor(train_idx)
    val_idx = train_idx
    test_idx = {
        'all': torch.LongTensor(np.hstack((normal_test, known_anomaly_test, unknown_anomaly_test))),
        'known': torch.LongTensor(np.hstack((normal_test, known_anomaly_test))),
        'unknown': torch.LongTensor(np.hstack((normal_test, unknown_anomaly_test))),
        'normal': torch.LongTensor(normal_test),
        'known_only': torch.LongTensor(known_anomaly_test),
        'unknown_only': torch.LongTensor(unknown_anomaly_test),
    }
    split_info = {
        'idx_train': train_idx,
        'idx_normal_train': normal_train,
        'idx_anomaly_train': known_anomaly_train,
        'idx_val': val_idx,
        'idx_test': test_idx
    }
    return split_info
def ad_split_num(labels, train_ratio, num_anomaly, class_info):
    known_anomaly = class_info['known_anomaly']
    unknown_anomaly_classes = class_info['unknown_anomaly']
    normal_classes = class_info['normal']
    print("normal_classes:{}".format(normal_classes))
    known_anomaly_idx = np.where(labels == known_anomaly)[0].flatten()
    normal_idx = select_class_idx_in_list(labels, normal_classes)
    unknown_anomaly_idx = select_class_idx_in_list(labels, unknown_anomaly_classes)
    normal_train, normal_rest = random_split(normal_idx, train_ratio)
    normal_val1, normal_rest2 = num_split(normal_rest, len(normal_train))
    normal_val2, normal_test = num_split(normal_rest2, len(normal_train))
    known_anomaly_train, known_anomaly_test = num_split(known_anomaly_idx, num_anomaly)
    unknown_anomaly_test = unknown_anomaly_idx
    train_idx = np.hstack((normal_train, known_anomaly_train))
    val1_idx = normal_val1
    val2_idx = normal_val2
    test_idx = {
        'all': torch.LongTensor(np.hstack((normal_test, known_anomaly_test, unknown_anomaly_test))),
        'known': torch.LongTensor(np.hstack((normal_test, known_anomaly_test))),
        'unknown': torch.LongTensor(np.hstack((normal_test, unknown_anomaly_test))),
        'normal': torch.LongTensor(normal_test),
        'known_only': torch.LongTensor(known_anomaly_test),
        'unknown_only': torch.LongTensor(unknown_anomaly_test),
    }
    split_info = {
        'idx_train': torch.LongTensor(train_idx),
        'idx_normal_train': normal_train,
        'idx_anomaly_train': known_anomaly_train,
        'idx_val1': torch.LongTensor(val1_idx),
        'idx_val2': torch.LongTensor(val2_idx),
        'idx_test': test_idx
    }
    return split_info
class DglDataset:
    def __init__(self, name='yelp', homo=True, anomaly_alpha=None, anomaly_std=None, view=None):
        name = name.lower()
        if name == 'yelp':
            dataset, _ = load_graphs('../dataset/yelp.bin')
        elif name == 'photo':
            dataset = AmazonCoBuyPhotoDataset()
        elif name == 'computer':
            dataset = AmazonCoBuyComputerDataset()
        elif name == 'cs':
            dataset = CoauthorCSDataset()
        elif name == 'tfinance':
            dataset, label_dict = load_graphs('../dataset/tfinance.bin')
        else:
            raise ValueError(f"Unsupported dataset: {name}")
        graph = dataset[0]
        if name == 'tfinance':
            graph.ndata['label'] = graph.ndata['label'].argmax(1)
        if isinstance(graph, dgl.DGLHeteroGraph) and name == 'yelp':
            if 'view' in locals():
                graph = dgl.edge_type_subgraph(graph, [view])
            else:
                raise ValueError("view must be defined for heterograph.")
        src, dst = graph.edges()
        if not torch.any(src == dst):
            graph = dgl.add_self_loop(graph)
        if name in ['tfinance', 'yelp', 'tsocial']:
            graph.ndata['label'], _ = load_labels(name)
        if 'label' in graph.ndata:
            graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        if name in ['yelp', 'tfinance', 'tsocial']:
            graph.ndata['feature'] = graph.ndata['feature'].float()
        else:
            graph.ndata['feature'] = graph.ndata['feat'].float()
        print(graph)
        self.graph = graph