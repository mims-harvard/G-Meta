import  torch, os
import  numpy as np
from    subgraph_data_processing import Subgraphs
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
        graphs_spt, labels_spt, graph_qry, labels_qry = map(list, zip(*samples))

        return graphs_spt, labels_spt, graph_qry, labels_qry

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    
    root = '../data/fold1/'

    config = [
        ('GraphConv', [args.input_dim, args.hidden_dim]),
        ('GraphConv', [args.hidden_dim, args.hidden_dim]),
        ('Linear', [args.hidden_dim, args.n_way])
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
    
    # batchsz here means total episode number
    db_train = Subgraphs(root, 'train', total_subgraph, info, center_node, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, batchsz=1000)
    db_val = Subgraphs(root, 'val', total_subgraph, info, center_node, n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=100)
    db_test = Subgraphs(root, 'test', total_subgraph, info, center_node, n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=100)

    for epoch in range(args.epoch//1000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(db_train, args.task_num, shuffle=True, num_workers=1, pin_memory=True, collate_fn = collate)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            
            # x_spt: a list of #task_num tasks, where each task is a mini-batch of k-shot * n_way subgraphs
            # y_spt: a list of #task_num lists of labels. Each list is of length k-shot * n_way int.

            #x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 30 == 0:
                print('epoch:', epoch, 'step:', step, '\ttraining acc:', accs)

            if step % 500 == 0:  # evaluation
                db_test = DataLoader(db_val, 1, shuffle=True, num_workers=1, pin_memory=True, collate_fn = collate)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    #x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), \
                    #                             x_qry.to(device), y_qry.to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('epoch:', epoch, 'Test acc:', accs)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=12)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--input_dim', type=int, help='input feature dim', default=1)
    argparser.add_argument('--hidden_dim', type=int, help='hidden dim', default=32)

    args = argparser.parse_args()

    main()
