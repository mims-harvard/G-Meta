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
from utils import sparse_mx_to_torch_sparse_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_regression(model, train_features, train_labels, idx_train, epochs, weight_decay, lr, adj):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features, adj)
        loss_train = F.cross_entropy(output[idx_train], train_labels[idx_train])
        loss_train.backward()
        optimizer.step()
    return model
#Train Model


def test_regression(model, test_features, test_labels, idx_test, adj):
    model.eval()
    output = model(test_features, adj)
    return accuracy(output[idx_test], test_labels[idx_test])
#Test Model

def reset_array():
    class1_train = []
    class2_train = []
    class1_test = []
    class2_test = []
    train_idx = []
    test_idx = []
#Clear Array


def main():
    args = get_citation_args()          #get args
    n_way = args.n_way                  #how many classes
    train_shot = args.train_shot        #train-shot
    test_shot = args.test_shot          #test-shot
    step = args.step                    
    node_num = args.node_num
    iteration = args.iteration

    accuracy_meta_test = []
    total_accuracy_meta_test = []

    set_seed(args.seed, args.cuda)
    #set seed

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

    node_num = adj.shape[0]
    class_label = list(np.unique(labels.cpu()))
    combination = list(combinations(class_label, 2))

    train_file = pd.read_csv(args.data_dir + '/fold' + str(args.fold_n) + '/train.csv')
    test_file = pd.read_csv(args.data_dir + '/fold' + str(args.fold_n) + '/test.csv')

    test_label = list(np.unique(test_file.label.values))
    train_label = list(np.unique(train_file.label.values))

    print('Train_Label_List {}: '.format(train_label))
    print('Test_Label_List {}: '.format(test_label))
    model = get_model(args.model, features.size(1), n_way, args.hidden, args.dropout, args.cuda).to(device)
    #create model

    for j in range(iteration):
        labels_local = labels.clone().detach()
        select_class = random.sample(train_label, n_way)
        # KH: each task?
        class1_idx = []
        class2_idx = []
        # KH: here we assume two ways.
        
        # KH: here it tries to create a task dataset label, and then assign all nodes corresponding to these labels to a set
        for k in range(node_num):
            if(labels_local[k] == select_class[0]):
                class1_idx.append(k)
                labels_local[k] = 0
            elif(labels_local[k] == select_class[1]):
                class2_idx.append(k)
                labels_local[k] = 1
                
        # KH: here it performs inner loop update        
        for m in range(step):
            # sample k shots for each class label in this specific task
            class1_train = random.sample(class1_idx,train_shot)
            class2_train = random.sample(class2_idx,train_shot)
            class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
            class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
            train_idx = class1_train + class2_train
            random.shuffle(train_idx)
            test_idx = class1_test + class2_test
            random.shuffle(test_idx)

            model = train_regression(model, features, labels_local, train_idx, args.epochs, args.weight_decay, args.lr, adj)
            acc_query = test_regression(model, features, labels_local, test_idx, adj)
            reset_array()
        print('training query accuracy:', acc_query)

    torch.save(model.state_dict(), 'model.pkl')
    #save model as 'model.pkl'

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

    for m in range(step):
        class1_train = random.sample(class1_idx, test_shot)
        class2_train = random.sample(class2_idx, test_shot)
        class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
        class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
        train_idx = class1_train + class2_train
        random.shuffle(train_idx)
        test_idx = class1_test + class2_test
        random.shuffle(test_idx)

        model_meta_trained = get_model(args.model, features.size(1), n_way, args.hidden, args.dropout, args.cuda).to(device)
        model_meta_trained.load_state_dict(torch.load('model.pkl'))
        #re-load model 'model.pkl'

        model_meta_trained = train_regression(model_meta_trained, features, labels_local, train_idx, args.epochs, args.weight_decay, args.lr, adj)
        acc_test = test_regression(model_meta_trained, features, labels_local, test_idx, adj)
        accuracy_meta_test.append(acc_test)
        total_accuracy_meta_test.append(acc_test)
        reset_array()
    print('testing accuracy: ', np.array(total_accuracy_meta_test).mean(axis=0).astype(np.float16))

if __name__ == '__main__':
    main()
