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
import time
import copy
import psutil

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
        graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx = map(list, zip(*samples))

        return graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx

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

    if args.graphlets == 'True':
        # create graphlets
        graphs = []
        for i in range(1, args.n_graphlets):
                graphs = graphs + generate_graphlet(i+1)

        print('There are {} number of graphlets'.format(len(graphs)))
        graphlets = dgl.batch(graphs)
    else:
        graphlets = 1
        # placeholder
        
    feat = np.load(root + 'features.npy', allow_pickle = True)

    with open(root + '/graph_dgl.pkl', 'rb') as f:
        dgl_graph = pickle.load(f)

    if args.task_setup == 'Disjoint':    
        with open(root + 'label.pkl', 'rb') as f:
            info = pickle.load(f)
    elif args.task_setup == 'Shared':
        if args.task_mode == 'True':
            root = root + '/task' + str(args.task_n) + '/'
        with open(root + 'label.pkl', 'rb') as f:
            info = pickle.load(f)

    total_class = len(np.unique(np.array(list(info.values()))))
    print('there are {} classes '.format(total_class))

    if args.task_setup == 'Disjoint':
        labels_num = args.n_way
    elif args.task_setup == 'Shared':
        labels_num = total_class

    if len(feat.shape) == 2:
        # single graph, to make it compatible to multiple graph retrieval.
        feat = [feat]    

    if args.graphlets == 'True':
        config = [
            ('GraphConv', [feat[0].shape[1], args.hidden_dim]),
            ('GraphConv', [args.hidden_dim, args.hidden_dim]),
            ('Attention', [args.hidden_dim, args.attention_size, args.hidden_dim, labels_num, len(graphs)]),
            ('Graphlet', [1, args.hidden_dim])
        ]
    else:
        config = [
            ('GraphConv', [feat[0].shape[1], args.hidden_dim]),
            ('GraphConv', [args.hidden_dim, args.hidden_dim]),
            ('Linear', [args.hidden_dim, labels_num])
        ]
    
    if args.link_pred_mode == 'True':
        config.append(('LinkPred', [True]))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    
    max_acc = 0
    model_max = copy.deepcopy(maml)
    
    root = root + 'fold' + str(args.fold_n) + '/'
    # batchsz here means total episode number
    db_train = Subgraphs(root, 'train', info, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, batchsz=args.batchsz, args = args, adjs = dgl_graph)
    db_val = Subgraphs(root, 'val', info, n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=100, args = args, adjs = dgl_graph)
    db_test = Subgraphs(root, 'test', info, n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=100, args = args, adjs = dgl_graph)
    
    
    if args.no_finetune==True:
        for epoch in range(args.epoch):
            # each episode(epoch) consists of 1000 batches, where each batch is a task, each task consists of support and query 
            db = DataLoader(db_train, args.task_num, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate)
            s_f = time.time()
            for step, (x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry) in enumerate(db):
                nodes_len = 0
                if step >= 1:
                    data_loading_time = time.time() - s_r
                else:
                    data_loading_time = time.time() - s_f
                s = time.time()
                # x_spt: a list of #task_num tasks, where each task is a mini-batch of k-shot * n_way subgraphs
                # y_spt: a list of #task_num lists of labels. Each list is of length k-shot * n_way int.                
                nodes_len += sum([sum([len(j) for j in i]) for i in n_spt])
                accs = maml(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat, graphlets)
                
                if step % args.train_result_report_steps == 0:
                    print('epoch:', epoch, 'step:', step, '\ttraining acc:', accs, '\ttime elapsed:', time.time() - s, '\tdata loading takes:', data_loading_time)
                    #print(psutil.virtual_memory())
                    #print('# of nodes for subgraphs in this step: ', nodes_len)
                if (step + 1) % args.val_result_report_steps == 0:  # evaluation
                    db_v = DataLoader(db_val, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate)
                    accs_all_test = []

                    for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry in db_v:
                        
                        accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat, graphlets)
                        accs_all_test.append(accs)

                    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                    print('epoch:', epoch, 'Val acc:', accs)
                    if accs[-1] > max_acc:
                        max_acc = accs[-1]
                        model_max = copy.deepcopy(maml)
                        
                s_r = time.time()
                
    db_t = DataLoader(db_test, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate)
    accs_all_test = []

    for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry in db_t:
        accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat, graphlets)
        accs_all_test.append(accs)

    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    print('Test acc:', accs)

    for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry in db_t:
        accs = model_max.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat, graphlets)
        accs_all_test.append(accs)
    torch.save(model_max.state_dict(), './model.pt')

    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    print('Early Stopped Test acc:', accs)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=10)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=24)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--input_dim', type=int, help='input feature dim', default=1)
    argparser.add_argument('--hidden_dim', type=int, help='hidden dim', default=64)
    argparser.add_argument('--attention_size', type=int, help='dim of attention_size', default=32)
    argparser.add_argument('--n_graphlets', type=int, help='up to n number of nodes in the graphlets', default=5)
    argparser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir.")
    argparser.add_argument('--fold_n', type=int, help='fold number', default=1)
    argparser.add_argument("--no_finetune", default=True, type=str, required=False, help="no finetune mode.")
    argparser.add_argument("--graphlets", default=False, type=str, required=False, help="use graphlets mode")
    argparser.add_argument("--task_setup", default='Disjoint', type=str, required=True, help="Select from Disjoint or Shared Setup. For Disjoint-Label, single/multiple graphs are both considered.")
    argparser.add_argument("--method", default='ProtoMAML', type=str, required=True, help="Select from ProtoMAML, MAML, ProtoNet")
    argparser.add_argument('--task_n', type=int, help='task number', default=1)
    argparser.add_argument("--task_mode", default='False', type=str, required=False, help="For Evaluating on Tasks")
    argparser.add_argument("--val_result_report_steps", default=100, type=int, required=False, help="validation report")
    argparser.add_argument("--train_result_report_steps", default=30, type=int, required=False, help="training report")
    argparser.add_argument("--num_workers", default=0, type=int, required=False, help="num of workers")
    argparser.add_argument("--batchsz", default=1000, type=int, required=False, help="batch size")
    argparser.add_argument("--link_pred_mode", default='False', type=str, required=False, help="For Link Prediction")

    args = argparser.parse_args()

    main()
