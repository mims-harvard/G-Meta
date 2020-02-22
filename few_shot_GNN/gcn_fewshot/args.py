import torch
import argparse


def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_shot', type=int, default=20, help='How many shot during meta-train')
    parser.add_argument('--test_shot', type=int, default=1, help='How many shot during meta-test')
    parser.add_argument('--n_way', type=int, default=2, help='Classes want to be classify')
    parser.add_argument('--step', type=int, default=50, help='How many times to random select node to test')
    parser.add_argument('--node_num', type=int, default=2708, help='Node number (dataset)')
    parser.add_argument('--iteration', type=int, default=50, help='Iteration each cross_validation')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset to use.')
    parser.add_argument('--model', type=str, default='GCN', help='Model to use.')
    parser.add_argument('--normalization', type=str, default='FirstOrderGCN', help='Normalization method for the adjacency matrix.')
    parser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')

    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
