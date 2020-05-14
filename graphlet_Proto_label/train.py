import  torch, os
import  numpy as np
from    subgraph_data_processing import Subgraphs
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

import networkx as nx
import numpy as np
from scipy.special import comb
from itertools import combinations 
import networkx.algorithms.isomorphism as iso
from tqdm import tqdm
import dgl

from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
        graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry = map(list, zip(*samples))

        return graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry

# helper function to create any number of graphlets

def generate_graphlet(n):
    non_iso_graph = []
    non_iso_graph_adj = []
    dgl_graph = []
    for i in tqdm(range(n-1, int(comb(n, 2))+1)):
    # for each of these possible # of edges
        arr = np.array(range(int((n**2-n)/2)))
        all_comb = list(combinations(arr, i)) 
        # all possible combination of edge positions 
        indices = np.triu_indices(n, 1)
        for m in range(len(all_comb)):
            # iterate over all these graphs
            adj = np.zeros((n,n))
            adj[indices[0][np.array(all_comb[m])], indices[1][np.array(all_comb[m])]] = 1
            adj_temp = adj
            adj = adj + adj.T
            #print(adj)
            if sum(np.sum(adj_temp, axis = 0) == 0) == 1:
                #the graph has to be connected
                new_graph = nx.from_numpy_matrix(adj)
                if len(non_iso_graph) == 0:
                    non_iso_graph.append(new_graph)
                    non_iso_graph_adj.append(adj)
                    S = dgl.DGLGraph()
                    S.from_networkx(new_graph)
                    dgl_graph.append(S)
                else:
                    is_iso = False
                    for g in non_iso_graph:
                        if iso.is_isomorphic(g, new_graph):
                            #print('yes')
                            is_iso = True
                            break
                    if not is_iso:
                        # not isomorphic to any of the current graphs
                        non_iso_graph.append(new_graph)
                        non_iso_graph_adj.append(adj)
                        
                        S = dgl.DGLGraph()
                        S.from_networkx(new_graph)
                        dgl_graph.append(S)
                        
    
    print('There are {} non-isomorphic graphs'.format(len(non_iso_graph)))
    return dgl_graph

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    
    root = args.data_dir
    # create graphlets
    graphs = []
    for i in range(1, args.n_graphlets):
            graphs = graphs + generate_graphlet(i+1)

    print('There are {} number of graphlets'.format(len(graphs)))
    graphlets = dgl.batch(graphs)


    config = [
        ('GraphConv', [args.input_dim, args.hidden_dim]),
        ('GraphConv', [args.hidden_dim, args.hidden_dim]),
        ('Attention', [args.hidden_dim, args.attention_size, args.hidden_dim, args.n_way, len(graphs)])
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    
    with open(root + 'list_subgraph.pkl', 'rb') as f:
        total_subgraph = pickle.load(f)
    
    with open(root + 'label.pkl', 'rb') as f:
        info = pickle.load(f)

    with open(root + 'center.pkl', 'rb') as f:
        center_node = pickle.load(f)  
    
    root = root + 'fold' + str(args.fold_n) + '/'
    # batchsz here means total episode number
    db_train = Subgraphs(root, 'train', total_subgraph, info, center_node, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, batchsz=1000)
    db_val = Subgraphs(root, 'val', total_subgraph, info, center_node, n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=100)
    db_test = Subgraphs(root, 'test', total_subgraph, info, center_node, n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=100)

    if args.no_finetune==True:
        for epoch in range(args.epoch):
            # fetch meta_batchsz num of episode each time

            # each episode(epoch) consists of 1000 batches, where each batch is a task, each task consists of support and query 
            db = DataLoader(db_train, args.task_num, shuffle=True, num_workers=1, pin_memory=True, collate_fn = collate)

            for step, (x_spt, y_spt, x_qry, y_qry, c_spt, c_qry) in enumerate(db):
                
                # x_spt: a list of #task_num tasks, where each task is a mini-batch of k-shot * n_way subgraphs
                # y_spt: a list of #task_num lists of labels. Each list is of length k-shot * n_way int.

                #x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
                
                accs = maml(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, graphlets)

                if step % 30 == 0:
                    print('epoch:', epoch, 'step:', step, '\ttraining acc:', accs)

                if step % 500 == 0:  # evaluation
                    db_v = DataLoader(db_val, 1, shuffle=True, num_workers=1, pin_memory=True, collate_fn = collate)
                    accs_all_test = []

                    for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry in db_v:
                        #x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), \
                        #                             x_qry.to(device), y_qry.to(device)

                        accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, graphlets)
                        accs_all_test.append(accs)

                    # [b, update_step+1]
                    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                    print('epoch:', epoch, 'Val acc:', accs)
        
    db_t = DataLoader(db_test, 1, shuffle=True, num_workers=1, pin_memory=True, collate_fn = collate)
    accs_all_test = []

    for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry in db_t:
        accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, graphlets)
        accs_all_test.append(accs)

    # [b, update_step+1]
    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    print('Test acc:', accs)



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=12)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--input_dim', type=int, help='input feature dim', default=1)
    argparser.add_argument('--hidden_dim', type=int, help='hidden dim', default=32)
    argparser.add_argument('--attention_size', type=int, help='dim of attention_size', default=32)
    argparser.add_argument('--n_graphlets', type=int, help='up to n number of nodes in the graphlets', default=5)
    argparser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir.")
    argparser.add_argument('--fold_n', type=int, help='fold number', default=1)
    argparser.add_argument("--no_finetune", default=True, type=str, required=False, help="no finetune mode.")

    args = argparser.parse_args()

    main()
