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


from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, g, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        """model parameters setting
        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError
        self.g = g

    def forward(self, h):
        # list of hidden representation at each layer (including input)
        #hidden_rep = [h]
        g = self.g

        h = g.in_degrees().view(-1, 1).float()
        #h = torch.tensor([1.]*g.number_of_nodes()).view(-1, 1).float()
        
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            #hidden_rep.append(h)

        #score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        #for i, h in enumerate(hidden_rep):
        #    pooled_h = self.pool(g, h)
        #    score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

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
    iteration = 100

    G = nx.from_numpy_matrix(np.load(args.data_dir + 'adj.npy'))
    adj = nx.adjacency_matrix(G)
    #adj = aug_normalized_adjacency(A)

    features = np.load(args.data_dir + 'features.npy')
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

    step = 10
    total_accuracy_meta_test = []
    accuracy_meta_test = []
    
    print('Train_Label_List {}: '.format(train_label))
    print('Test_Label_List {}: '.format(test_label))

    model = GIN(g = g, num_layers = 2, num_mlp_layers = 3,
        input_dim = 1, hidden_dim = 32, output_dim = 2, final_dropout = 0.5,
        learn_eps = True, graph_pooling_type = "mean", neighbor_pooling_type = "max")
    
    if cuda:
        model.cuda()

    for j in range(iteration):
        random.seed(j)
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
            #print('Train query accuracy one class combo: ', acc_query)
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

    for m in range(100): # average from 100 datapoints
        random.seed(m)
        class1_train = random.sample(class1_idx, test_shot)
        class2_train = random.sample(class2_idx, test_shot)
        class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
        class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
        train_idx = class1_train + class2_train
        random.shuffle(train_idx)
        test_idx = class1_test + class2_test
        random.shuffle(test_idx)

        model_meta_trained = GIN(g = g, num_layers = 2, num_mlp_layers = 3,
        input_dim = 1, hidden_dim = 32, output_dim = 2, final_dropout = 0.1,
        learn_eps = True, graph_pooling_type = "mean", neighbor_pooling_type = "max")
    
        if cuda:
            model_meta_trained.cuda()
        model_meta_trained.load_state_dict(torch.load('model.pkl'))
       
        accs = []

        for i in range(step):
            model_meta_trained = train_regression(model_meta_trained, features, labels_local, train_idx, args.n_epochs, args.weight_decay, args.lr)
            acc_test = test_regression(model_meta_trained, features, labels_local, test_idx)
            accs.append(acc_test)
        total_accuracy_meta_test.append(accs)
        reset_array()

    #print(total_accuracy_meta_test)
    print('Test query accuracy: ', np.mean(np.array(total_accuracy_meta_test), axis = 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
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

        