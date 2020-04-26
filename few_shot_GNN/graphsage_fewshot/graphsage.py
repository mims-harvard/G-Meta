import argparse
import random
import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl
from dgl.data import register_data_args, load_data
import dgl.function as fn
from itertools import combinations
import networkx as nx
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Aggregator(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation=None, bias=True):
        super(Aggregator, self).__init__()
        self.g = g
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)  # (F, EF) or (2F, EF)
        self.activation = activation
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, node):
        nei = node.mailbox['m']  # (B, N, F)
        h = node.data['h']  # (B, F)
        h = self.concat(h, nei, node)  # (B, F) or (B, 2F)
        h = self.linear(h)   # (B, EF)
        if self.activation:
            h = self.activation(h)
        norm = torch.pow(h, 2)
        norm = torch.sum(norm, 1, keepdim=True)
        norm = torch.pow(norm, -0.5)
        norm[torch.isinf(norm)] = 0
        # h = h * norm
        return {'h': h}

    @abc.abstractmethod
    def concat(self, h, nei, nodes):
        raise NotImplementedError


class MeanAggregator(Aggregator):
    def __init__(self, g, in_feats, out_feats, activation, bias):
        super(MeanAggregator, self).__init__(g, in_feats, out_feats, activation, bias)

    def concat(self, h, nei, nodes):
        degs = self.g.in_degrees(nodes.nodes()).float()
        if h.is_cuda:
            degs = degs.cuda(h.device)
        concatenate = torch.cat((nei, h.unsqueeze(1)), 1)
        concatenate = torch.sum(concatenate, 1) / degs.unsqueeze(1)
        return concatenate  # (B, F)


class PoolingAggregator(Aggregator):
    def __init__(self, g, in_feats, out_feats, activation, bias):  # (2F, F)
        super(PoolingAggregator, self).__init__(g, in_feats*2, out_feats, activation, bias)
        self.mlp = PoolingAggregator.MLP(in_feats, in_feats, F.relu, False, True)

    def concat(self, h, nei, nodes):
        nei = self.mlp(nei)  # (B, F)
        concatenate = torch.cat((nei, h), 1)  # (B, 2F)
        return concatenate

    class MLP(nn.Module):
        def __init__(self, in_feats, out_feats, activation, dropout, bias):  # (F, F)
            super(PoolingAggregator.MLP, self).__init__()
            self.linear = nn.Linear(in_feats, out_feats, bias=bias)  # (F, F)
            self.dropout = nn.Dropout(p=dropout)
            self.activation = activation
            nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

        def forward(self, nei):
            nei = self.dropout(nei)  # (B, N, F)
            nei = self.linear(nei)
            if self.activation:
                nei = self.activation(nei)
            max_value = torch.max(nei, dim=1)[0]  # (B, F)
            return max_value


class GraphSAGELayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 aggregator_type,
                 bias=True,
                 ):
        super(GraphSAGELayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(p=dropout)
        if aggregator_type == "pooling":
            self.aggregator = PoolingAggregator(g, in_feats, out_feats, activation, bias)
        else:
            self.aggregator = MeanAggregator(g, in_feats, out_feats, activation, bias)

    def forward(self, h):
        h = self.dropout(h)
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_src(src='h', out='m'), self.aggregator)
        h = self.g.ndata.pop('h')
        return h


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphSAGELayer(g, in_feats, n_hidden, activation, dropout, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphSAGELayer(g, n_hidden, n_hidden, activation, dropout, aggregator_type))
        # output layer
        self.layers.append(GraphSAGELayer(g, n_hidden, n_classes, None, dropout, aggregator_type))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def reset_array():
    class1_train = []
    class2_train = []
    class1_test = []
    class2_test = []
    train_idx = []
    test_idx = []


def train_regression(model, features, labels_local, train_idx, epochs, weight_decay, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fcn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        logits = model(features)
        #print(logits)
        loss = loss_fcn(logits[train_idx], labels_local[train_idx])
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def test_regression(model, test_features, test_labels, idx_test):
    model.eval()
    return evaluate(model, test_features, test_labels, idx_test)


def main(args):
    train_shot = args.train_shot
    test_shot = args.test_shot
    iteration = 15

    G = nx.from_numpy_matrix(np.load(args.data_dir + 'graph_adj.npy'))
    adj = nx.adjacency_matrix(G)
    #adj = aug_normalized_adjacency(A)

    features = np.eye(adj.shape[0])
    labels = pd.read_csv(args.data_dir + 'data.csv')
    labels = labels.label.values
     
    features = torch.FloatTensor(features).float().to(device)
    labels = torch.LongTensor(labels).to(device)

    node_num = adj.shape[0]
    class_label = list(np.unique(labels.cpu()))
    combination = list(combinations(class_label, 2))

    train_file = pd.read_csv(args.data_dir + '/fold' + str(args.fold_n) + '/train.csv')
    test_file = pd.read_csv(args.data_dir + '/fold' + str(args.fold_n) + '/test.csv')

    test_label = list(np.unique(test_file.label.values))
    train_label = list(np.unique(train_file.label.values))

    print('Train_Label_List {}: '.format(train_label))
    print('Test_Label_List {}: '.format(test_label))

    in_feats = features.shape[1]
    n_classes = 2

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        print("use cuda:", args.gpu)

    g = dgl.DGLGraph()
    g.from_networkx(G)

    step = 50
    total_accuracy_meta_test = []
    accuracy_meta_test = []
    
    print('Train_Label_List {}: '.format(train_label))
    print('Test_Label_List {}: '.format(test_label))
    model = GraphSAGE(g, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.aggregator_type)
    if cuda:
        model.cuda()

    for j in range(iteration):
        labels_local = labels.clone().detach()
        select_class = random.sample(train_label, n_classes)
        print('ITERATION {} Train_Label: {}'.format(j+1, select_class))
        class1_idx = []
        class2_idx = []
        for k in range(node_num):
            if(labels_local[k] == select_class[0]):
                class1_idx.append(k)
                labels_local[k] = 0
            elif(labels_local[k] == select_class[1]):
                class2_idx.append(k)
                labels_local[k] = 1
        for m in range(50):
            random.seed(m)
            class1_train = random.sample(class1_idx, train_shot)
            class2_train = random.sample(class2_idx, train_shot)
            #print(class1_train)
            #print(class2_train)
            class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
            class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
            train_idx = class1_train + class2_train
            random.shuffle(train_idx)
            test_idx = class1_test + class2_test
            random.shuffle(test_idx)
            model = train_regression(model, features, labels_local, train_idx, args.n_epochs, args.weight_decay, args.lr)
            acc_query = test_regression(model, features, labels_local, test_idx)
            accuracy_meta_test.append(acc_query)
            print('Train query accuracy one class combo: ', acc_query)
            reset_array()
    print('Meta-Train_Accuracy: {}'.format(torch.tensor(accuracy_meta_test).numpy().mean()))
    accuracy_meta_test = []
    torch.save(model.state_dict(), 'model.pkl')

    labels_local = labels.clone().detach()
    select_class = random.sample(test_label, 2)
    print('Test_Label {}: '.format(select_class))
    class1_idx = []
    class2_idx = []
    reset_array()
    for k in range(node_num):
        if (labels_local[k] == select_class[0]):
            class1_idx.append(k)
            labels_local[k] = 0
        elif (labels_local[k] == select_class[1]):
            class2_idx.append(k)
            labels_local[k] = 1
    for m in range(step):
        random.seed(m)
        class1_train = random.sample(class1_idx, test_shot)
        class2_train = random.sample(class2_idx, test_shot)
        class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
        class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
        train_idx = class1_train + class2_train
        random.shuffle(train_idx)
        test_idx = class1_test + class2_test
        random.shuffle(test_idx)

        model_meta_trained = GraphSAGE(g, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.aggregator_type)
        if cuda:
            model_meta_trained.cuda()
        model_meta_trained.load_state_dict(torch.load('model.pkl'))

        model_meta_trained = train_regression(model_meta_trained, features, labels_local, train_idx, args.n_epochs, args.weight_decay, args.lr)
        acc_test = test_regression(model_meta_trained, features, labels_local, test_idx)
        accuracy_meta_test.append(acc_test)
        total_accuracy_meta_test.append(acc_test)
        reset_array()
    print('Test query accuracy one class combo: ', total_accuracy_meta_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=5,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Weight for L2 loss")
    parser.add_argument('--train_shot', type=int, default=1, help='How many shot during meta-train')
    parser.add_argument('--test_shot', type=int, default=1, help='How many shot during meta-test')
    parser.add_argument('--data_dir', type=str, help='Dataset to use.')
    parser.add_argument('--fold_n', type=int, default=1, help='fold number')
    args = parser.parse_args()
    print(args)

    main(args)

        