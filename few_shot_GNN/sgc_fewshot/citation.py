import torch
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations
import random
import networkx as nx
import numpy as np
import pandas as pd
from args import get_citation_args
from utils import load_citation, set_seed
from models import get_model
from metrics import accuracy
from normalization import aug_normalized_adjacency, row_normalize
from utils import sparse_mx_to_torch_sparse_tensor, sgc_precompute, set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_regression(model, train_features, train_labels, epochs, weight_decay, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    return model


def test_regression(model, test_features, test_labels):
    model.eval()
    output = model(test_features)
    return accuracy(output, test_labels)


def reset_array():
    class1_train = []
    class2_train = []
    class1_test = []
    class2_test = []
    train_idx = []
    test_idx = []


def main():
    args = get_citation_args()
    n_way = args.n_way
    train_shot = args.train_shot
    test_shot = args.test_shot
    step = args.step
    node_num = args.node_num
    iteration = args.iteration

    accuracy_meta_test = []
    total_accuracy_meta_test = []

    set_seed(args.seed, args.cuda)

    G = nx.from_numpy_matrix(np.load(args.data_dir + 'graph_adj.npy'))
    A = nx.adjacency_matrix(G)
    adj = aug_normalized_adjacency(A)

    features = np.eye(adj.shape[0])
    features = row_normalize(features)

    labels = pd.read_csv(args.data_dir + 'data.csv')
    labels = labels.label.values

    adj = sparse_mx_to_torch_sparse_tensor(adj).float().to(device)
     
    features = torch.FloatTensor(features).float().to(device)
    labels = torch.LongTensor(labels).to(device)
    
    accuracy_meta_test = []

    node_num = adj.shape[0]
    class_label = list(np.unique(labels.cpu()))
    combination = list(combinations(class_label, 2))

    train_file = pd.read_csv(args.data_dir + '/fold' + str(args.fold_n) + '/train.csv')
    test_file = pd.read_csv(args.data_dir + '/fold' + str(args.fold_n) + '/test.csv')

    test_label = list(np.unique(test_file.label.values))
    train_label = list(np.unique(train_file.label.values))

    if args.model == 'SGC':
        features = sgc_precompute(features, adj, args.degree)

    print('Train_Label_List {}: '.format(train_label))
    print('Test_Label_List {}: '.format(test_label))
    model = get_model(args.model, features.size(1), n_way, args.cuda)

    for j in range(iteration):
        random.seed(j)
        labels_local = labels.clone().detach()
        select_class = random.sample(train_label, n_way)
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
        for m in range(step):
            random.seed(m)
            class1_train = random.sample(class1_idx,train_shot)
            class2_train = random.sample(class2_idx,train_shot)
            class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
            class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
            train_idx = class1_train + class2_train
            random.shuffle(train_idx)
            test_idx = class1_test + class2_test
            random.shuffle(test_idx)

            model = train_regression(model, features[train_idx], labels_local[train_idx], args.epochs, args.weight_decay, args.lr)
            acc_query = test_regression(model, features[test_idx], labels_local[test_idx])
            accuracy_meta_test.append(acc_query)
            reset_array()
        
    print('Meta-Train_Accuracy: {}'.format(torch.tensor(accuracy_meta_test).numpy().mean()))

    torch.save(model.state_dict(), 'model.pkl')

    labels_local = labels.clone().detach()
    select_class = random.sample(test_label, 2)
    class1_idx = []
    class2_idx = []
    reset_array()
    for k in range(node_num):
        if(labels_local[k] == select_class[0]):
            class1_idx.append(k)
            labels_local[k] = 0
        elif(labels_local[k] == select_class[1]):
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

        model_meta_trained = get_model(args.model, features.size(1), n_way, args.cuda)
        model_meta_trained.load_state_dict(torch.load('model.pkl'))

        
        accs = []

        model_meta_trained = train_regression(model_meta_trained, features[train_idx], labels_local[train_idx], args.epochs, args.weight_decay, args.lr)
        acc_test = test_regression(model_meta_trained, features[test_idx], labels_local[test_idx])
        accs.append(acc_test)
        total_accuracy_meta_test.append(accs)
        reset_array()

    #print(total_accuracy_meta_test)
    print('Test query accuracy: ', np.mean(np.array(total_accuracy_meta_test), axis = 0))


if __name__ == '__main__':
    main()
