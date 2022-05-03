import numpy as np
import scipy.sparse as sp
import torch
import os
import pickle
import networkx as nx


# MUTAG数据集特征，188个图，总共3371个结点，7442条边，为无向图

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_syn_data(path="datas/BA_shapes", dataset='BA_shapes'):
    print('Loading synthetic {} dataset...'.format(dataset))
    with open(os.path.join(path, dataset + '.pkl'), 'rb') as f:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pickle.load(f)
    length = 1
    adj_list = torch.tensor(adj, dtype=torch.float32).unsqueeze(0)
    features_list = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    idx_train = torch.LongTensor(np.arange(length)[train_mask])
    idx_val = torch.LongTensor(np.arange(length)[val_mask])
    idx_test = torch.LongTensor(np.arange(length)[test_mask])

    y_train = y_train.argmax(-1).tolist()
    y_val = y_val.argmax(-1).tolist()
    y_test = y_test.argmax(-1).tolist()

    graph_labels = np.zeros(length)
    for i, idx in enumerate(idx_train):
        graph_labels[idx] = y_train[i]
    for i, idx in enumerate(idx_val):
        graph_labels[idx] = y_val[i]
    for i, idx in enumerate(idx_test):
        graph_labels[idx] = y_test[i]
    graph_labels = torch.LongTensor(graph_labels)

    return adj_list, features_list, graph_labels, idx_train, idx_test, idx_val


def load_split_BA_2Motifs(path='datas/BA_2Motifs', split_train=0.8, split_val=0.1):
    print('Loading synthetic BA_2Motifs dataset...')
    with open(os.path.join(path, "BA_2Motifs.pkl"), 'rb') as f:
        raw_dense_edges, raw_node_features, raw_graph_labels = pickle.load(f)
    adj_list = []
    features_list = []
    graph_labels = []
    for n in range(raw_dense_edges.shape[0]):
        features_list.append(raw_node_features[n])
        adj_list.append(raw_dense_edges[n])
        graph_labels.append(np.where(raw_graph_labels[n])[0][0])

    num_total = raw_dense_edges.shape[0]
    num_train = int(split_train * num_total)
    num_val = int((split_train + split_val) * num_total)
    if (num_train == num_val or num_val == num_total):
        return

    features_list = torch.FloatTensor(features_list)
    adj_list = torch.FloatTensor(adj_list)
    graph_labels = torch.LongTensor(graph_labels)

    idx_train = range(num_train)
    idx_val = range(num_train, num_val)
    idx_test = range(num_val, num_total)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # 返回值一次为 188个图的邻接矩阵列表  188个图的特征矩阵列表  188个图的label， 每个图的起始结点索引号， 训练集索引号，
    # 验证集索引号， 测试集索引号
    return adj_list, features_list, graph_labels, idx_train, idx_val, idx_test


def load_split_data_with_node_labels(path="datas/MUTAG/", dataset="MUTAG", split_train=0.8, split_val=0.1):
    """Load MUTAG data """
    if not path.endswith('/'):
        path += '/'
    if not dataset.endswith('_'):
        dataset += '_'
    print('Loading {} dataset...'.format(dataset))

    # 加载图的标签
    graph_labels = np.genfromtxt("{}{}graph_labels.txt".format(path, dataset),
                                 dtype=np.dtype(int))
    graph_labels = encode_onehot(graph_labels)  # (188, 2)
    graph_labels = torch.LongTensor(np.where(graph_labels)[1])  # (188, 1)

    # 图结点的索引号
    graph_idx = np.genfromtxt("{}{}graph_indicator.txt".format(path, dataset),
                              dtype=np.dtype(int))

    graph_idx = np.array(graph_idx, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(graph_idx)}  # key, value表示第key个图的起始结点索引号为value
    length = len(idx_map.keys())  # 总共有多少个图
    num_nodes = [idx_map[n] - idx_map[n - 1] if n - 1 > 1 else idx_map[n] for n in
                 range(1, length + 1)]  # 一个长度188的list，表示没个图有多少个结点
    max_num_nodes = max(num_nodes)  # 最大的一个图有多少个结点
    features_list = []
    adj_list = []
    prev = 0

    # 结点的标签
    nodeidx_features = np.genfromtxt("{}{}node_labels.txt".format(path, dataset), delimiter=",",
                                     dtype=np.dtype(int))
    node_features = np.zeros((nodeidx_features.shape[0], max(nodeidx_features) + 1))
    node_features[np.arange(nodeidx_features.shape[0]), nodeidx_features] = 1

    # 边信息
    edges_unordered = np.genfromtxt("{}{}A.txt".format(path, dataset), delimiter=",",
                                    dtype=np.int32)

    # 边的标签
    edges_label = np.genfromtxt("{}{}edge_labels.txt".format(path, dataset), delimiter=",",
                                dtype=np.int32)  # shape = (7442,)

    # 生成邻接矩阵A，该邻接矩阵包括了数据集中所有的边
    adj = sp.coo_matrix((edges_label, (edges_unordered[:, 0] - 1, edges_unordered[:, 1] - 1)))

    # 论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    node_features = normalize(node_features)
    adj = normalize(adj + sp.eye(adj.shape[0]))  # 对应公式A~=A+IN
    adj = adj.tocsr()

    for n in range(1, length + 1):
        # entry为第n个图的特征矩阵X
        entry = np.zeros((max_num_nodes, max(nodeidx_features) + 1))
        entry[:idx_map[n] - prev] = node_features[prev:idx_map[n]]
        entry = torch.FloatTensor(entry)
        features_list.append(entry.tolist())

        # entry为第n个图的邻接矩阵A
        entry = np.zeros((max_num_nodes, max_num_nodes))
        entry[:idx_map[n] - prev, :idx_map[n] - prev] = adj[prev:idx_map[n]].todense()[prev:idx_map[n]]
        entry = torch.FloatTensor(entry)
        adj_list.append(entry.tolist())

        prev = idx_map[n]  # prev为下个图起始结点的索引号

    num_total = max(graph_idx)
    num_train = int(split_train * num_total)
    num_val = int((split_train + split_val) * num_total)

    if (num_train == num_val or num_val == num_total):
        return

    features_list = torch.FloatTensor(features_list)
    adj_list = torch.FloatTensor(adj_list)

    idx_train = range(num_train)
    idx_val = range(num_train, num_val)
    idx_test = range(num_val, num_total)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # 返回值一次为 188个图的邻接矩阵列表  188个图的特征矩阵列表  188个图的label， 每个图的起始结点索引号， 训练集索引号，
    # 验证集索引号， 测试集索引号
    return adj_list, features_list, graph_labels, idx_map, idx_train, idx_val, idx_test


def load_split_data_with_node_attrs(path="datas/Graph-Twitter/", dataset="Graph-Twitter", split_train=0.8, split_val=0.1):
    """Load MUTAG data """
    if not path.endswith('/'):
        path += '/'
    if not dataset.endswith('_'):
        dataset += '_'
    print('Loading {} dataset...'.format(dataset))

    # 加载图的标签
    graph_labels = np.genfromtxt("{}{}graph_labels.txt".format(path, dataset),
                                 dtype=np.dtype(int))
    graph_labels = encode_onehot(graph_labels)  # (188, 2)
    graph_labels = torch.LongTensor(np.where(graph_labels)[1])  # (188, 1)

    # 图结点的索引号
    graph_idx = np.genfromtxt("{}{}graph_indicator.txt".format(path, dataset),
                              dtype=np.dtype(int))

    graph_idx = np.array(graph_idx, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(graph_idx)}  # key, value表示第key个图的起始结点索引号为value
    length = len(idx_map.keys())  # 总共有多少个图
    num_nodes = [idx_map[n] - idx_map[n - 1] if n - 1 > 1 else idx_map[n] for n in
                 range(1, length + 1)]  # 一个长度188的list，表示没个图有多少个结点
    max_num_nodes = max(num_nodes)  # 最大的一个图有多少个结点
    features_list = []
    adj_list = []
    prev = 0

    # 结点的标签
    with open('{}{}node_attributes.pkl'.format(path, dataset), 'rb') as f:
        node_features = pickle.load(f)

    # 边信息
    edges_unordered = np.genfromtxt("{}{}A.txt".format(path, dataset), delimiter=",",
                                    dtype=np.int32)
    # 边的标签
    edges_label = np.ones((edges_unordered.shape[0],), dtype=np.float32)

    # 生成邻接矩阵A，该邻接矩阵包括了数据集中所有的边
    M = node_features.shape[0]
    adj = sp.coo_matrix((edges_label, (edges_unordered[:, 0] - 1, edges_unordered[:, 1] - 1)),
                        shape=(M, M))

    # 论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    node_features = normalize(node_features)
    adj = normalize(adj + sp.eye(adj.shape[0], dtype=np.float32))  # 对应公式A~=A+IN
    adj = adj.tocsr()

    print('Creating entries...')
    for n in range(1, length + 1):
        # entry为第n个图的特征矩阵X
        print(n, '/', length)
        entry = np.zeros((max_num_nodes, node_features.shape[1]), dtype=np.float32)
        entry[:idx_map[n] - prev] = node_features[prev:idx_map[n]]
        entry = torch.FloatTensor(entry)
        features_list.append(entry.tolist())

        # entry为第n个图的邻接矩阵A
        entry = np.zeros((max_num_nodes, max_num_nodes), dtype=np.float32)
        entry[:idx_map[n] - prev, :idx_map[n] - prev] = adj[prev:idx_map[n]].todense()[:, prev:idx_map[n]]
        entry = torch.FloatTensor(entry)
        adj_list.append(entry.tolist())

        prev = idx_map[n]  # prev为下个图起始结点的索引号

    print('All entries loaded!')
    num_total = max(graph_idx)
    num_train = int(split_train * num_total)
    num_val = int((split_train + split_val) * num_total)

    if (num_train == num_val or num_val == num_total):
        return

    features_list = torch.FloatTensor(features_list)
    adj_list = torch.FloatTensor(adj_list)

    idx_train = range(num_train)
    idx_val = range(num_train, num_val)
    idx_test = range(num_val, num_total)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # 返回值一次为 188个图的邻接矩阵列表  188个图的特征矩阵列表  188个图的label， 每个图的起始结点索引号， 训练集索引号，
    # 验证集索引号， 测试集索引号
    print('Dataset loaded!')
    return adj_list, features_list, graph_labels, idx_map, idx_train, idx_val, idx_test
