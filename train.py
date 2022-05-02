import json

import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from Load_dataset import load_split_data_with_node_labels, load_syn_data, \
    load_split_BA_2Motifs, load_split_data_with_node_attrs, accuracy
from Model import GCN

import argparse

epochs = 1000
seed = 200
lr = 0.001
dropout = 0.0
weight_decay = 0.005

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class EarlyStopping():
    def __init__(self, patience=10, min_loss=0.5, hit_min_before_stopping=False):
        self.patience = patience
        self.counter = 0
        self.hit_min_before_stopping = hit_min_before_stopping
        if hit_min_before_stopping:
            self.min_loss = min_loss
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss:
            self.counter += 1
            if self.counter > self.patience:
                if self.hit_min_before_stopping == True and loss > self.min_loss:
                    print("Cannot hit mean loss, will continue")
                    self.counter -= self.patience
                else:
                    self.early_stop = True
        else:
            self.best_loss = loss
            counter = 0


if __name__ == '__main__':
    # adj_list: [188, 29, 29]
    # features_list: [188, 29, 7]
    # graph_labels: [188]
    parser = argparse.ArgumentParser(description='XGNN arguments.')
    parser.add_argument('--ds_path', type=str)
    parser.add_argument('--ds_name', type=str)
    parser.add_argument('--feat', type=str, choices=['node-feat', 'node-labels'])
    args = parser.parse_args()

    train_ratio = 0.8
    val_ratio = 0.1
    if args.feat == 'node-labels':
        adj_list, features_list, graph_labels, _, idx_train, idx_val, idx_test = load_split_data_with_node_labels(
            path=args.ds_path,
            dataset=args.ds_name,
            split_train=train_ratio,
            split_val=val_ratio
        )
    else:
        if args.ds_name.lower() == 'ba_2motifs':
            adj_list, features_list, graph_labels, idx_train, idx_val, idx_test = load_split_BA_2Motifs(
                path=args.ds_path,
                split_train=train_ratio,
                split_val=val_ratio
            )
        else:
            adj_list, features_list, graph_labels, _, idx_train, idx_val, idx_test = load_split_data_with_node_attrs(
                path=args.ds_path,
                dataset=args.ds_name,
                split_train=train_ratio,
                split_val=val_ratio
            )

    idx_train = torch.cat([idx_train, idx_val, idx_test])

    model = GCN(nfeat=features_list[0].shape[1],  # nfeat = 7
                nclass=graph_labels.max().item() + 1,  # nclass = 2
                dropout=dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    model
    features_list = features_list
    adj_list = adj_list
    graph_labels = graph_labels
    idx_train = idx_train
    idx_val = idx_val
    idx_test = idx_test

    # 训练模型
    early_stopping = EarlyStopping(10, hit_min_before_stopping=True)
    t_total = time.time()

    results = []
    best_epoch = None
    for epoch in range(epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        # # Split
        outputs = []
        for i in idx_train:
            output = model(features_list[i], adj_list[i])
            output = output.unsqueeze(0)
            outputs.append(output)
        output = torch.cat(outputs, dim=0)

        loss_train = F.cross_entropy(output, graph_labels[idx_train])
        acc_train = accuracy(output, graph_labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        outputs = []
        for i in idx_val:
            output = model(features_list[i], adj_list[i])
            output = output.unsqueeze(0)
            outputs.append(output)
        output = torch.cat(outputs, dim=0)
        loss_val = F.cross_entropy(output, graph_labels[idx_val])
        acc_val = accuracy(output, graph_labels[idx_val])

        model.eval()
        outputs = []
        for i in idx_test:
            output = model(features_list[i], adj_list[i])
            output = output.unsqueeze(0)
            outputs.append(output)
        output = torch.cat(outputs, dim=0)
        loss_test = F.cross_entropy(output, graph_labels[idx_test])
        acc_test = accuracy(output, graph_labels[idx_test])

        result = {
            'epoch': epoch,
            'loss_train': round(loss_train.item(), 4),
            'acc_train': round(acc_train.item()),
            'loss_val': round(loss_val.item(), 4),
            'acc_val': round(acc_val.item(), 4),
            'loss_test': round(loss_test.item(), 4),
            'acc_test': round(acc_test.item(), 4),
            'time': round(time.time() - t, 4)
        }
        print(result)
        if best_epoch is None:
            best_epoch = epoch
        else:
            if acc_val > results[best_epoch]['acc_val']:
                best_epoch = epoch
        results.append(result)

        early_stopping(loss_val)
        if early_stopping.early_stop == True:
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("Best result:", results[best_epoch])

    results = {
        'configs': {
            'epochs': epochs,
            'seed': seed,
            'lr': lr,
            'dropout': dropout,
            'weight_decay': weight_decay,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio
        },
        'best_epoch': best_epoch,
        'epoch_results': results
    }
    with open(f'results/{args.ds_name}.json', 'w') as f:
        json.dump(results, f)

    model_path = f'model/{args.ds_name}.pth'
    torch.save(model.state_dict(), model_path)
