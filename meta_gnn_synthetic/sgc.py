import torch
import numpy as np
import argparse
import pandas as pd

from itertools import combinations
from utils import load_citation, sgc_precompute, set_seed, sparse_mx_to_torch_sparse_tensor
from meta import Meta
from sgc_data_generator import sgc_data_generator
from scipy.sparse import csr_matrix, isspmatrix
import networkx as nx
from normalization import aug_normalized_adjacency, row_normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    step = args.step
    set_seed(args.seed)
    
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
    #print(adj)
    #adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    features = sgc_precompute(features, adj, args.degree)

    node_num = adj.shape[0]
    class_label = list(np.unique(labels.cpu()))
    combination = list(combinations(class_label, 2))

    config = [
        ('linear', [args.hidden, features.size(1)]),
        ('linear', [args.n_way, args.hidden])
    ]
    # KH: there are only two ways for this code

    # for each combination of test label as a task.
    train_file = pd.read_csv(args.data_dir + '/fold' + str(args.fold_n) + '/train.csv')
    test_file = pd.read_csv(args.data_dir + '/fold' + str(args.fold_n) + '/test.csv')

    maml = Meta(args, config).to(device)
    # for each CV fold, just get the first task of the combination as the test label.
    test_label = list(np.unique(test_file.label.values))
    train_label = list(np.unique(train_file.label.values))
    print('Train_Label_List: {} '.format(train_label))
    print('Test_Label_List: {} '.format(test_label))

    for j in range(args.epoch):
        # KH: for each episode, sample tasks
        x_spt, y_spt, x_qry, y_qry = sgc_data_generator(features, labels, node_num, train_label, args.task_num, args.n_way, args.k_spt, args.k_qry)
        accs = maml.forward(x_spt, y_spt, x_qry, y_qry)
        
        if j % 100 == 0: 
            print('Step:', j, '\tMeta_Training_Accuracy:', accs)
        if j % 1000 == 0:
            # for every 100 steps, validate it. For testing, we need an additional set.
            torch.save(maml.state_dict(), 'maml.pkl')
            meta_test_acc = []
            for k in range(step):
                model_meta_trained = Meta(args, config).to(device)
                model_meta_trained.load_state_dict(torch.load('maml.pkl'))
                model_meta_trained.eval()
                x_spt, y_spt, x_qry, y_qry = sgc_data_generator(features, labels, node_num, test_label, args.task_num, args.n_way, args.k_spt, args.k_qry)
                accs = model_meta_trained.forward(x_spt, y_spt, x_qry, y_qry)
                meta_test_acc.append(accs)
            print('Step: {}, Meta-Test_Accuracy: {}'.format(j, np.array(meta_test_acc).mean(axis=0).astype(np.float16)))


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.003)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.5)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=12)
    argparser.add_argument('--hidden', type=int, help='Number of hidden units', default=16)

    argparser.add_argument('--dataset', type=str, default='citeseer', help='Dataset to use.')
    argparser.add_argument('--normalization', type=str, default='AugNormAdj', help='Normalization method for the adjacency matrix.')
    argparser.add_argument('--seed', type=int, default=42, help='Random seed.')
    argparser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')
    argparser.add_argument('--step', type=int, default=50, help='How many times to random select node to test')
    argparser.add_argument('--data_dir', type=str, help='Dataset to use.')
    argparser.add_argument('--fold_n', type=int, default=1, help='fold number')

    args = argparser.parse_args()

    main(args)
