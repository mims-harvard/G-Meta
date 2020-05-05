import argparse
import os
import random
import numpy as np
import torch

from code.data_generator import DataGenerator
from code.maml import meta_gradient_step
from code.models import GCN, GCN_Proto, GCN_Structure

##############
# Parameters #
##############
parser = argparse.ArgumentParser(description='graph transfer')
parser.add_argument("--datapath", default="../data/", type=str)
parser.add_argument("--graphpath", default="../data/graph/", type=str)
parser.add_argument('--logdir', type=str, default='/tmp/data', help='directory for summaries and checkpoints.')

# data generate hyperparameters
parser.add_argument("--random_seed", default='1', type=int)
parser.add_argument("--batch_n", default=5, type=int)
parser.add_argument('--A_n', default=18872 + 1, type=int)
parser.add_argument('--P_n', default=12334 + 1, type=int)
parser.add_argument('--V_n', default=18, type=int)
parser.add_argument("--in_f_d", default=128, type=int)
parser.add_argument("--sample_g_n", default=100, type=int)
parser.add_argument("--test_sample_g_n", default=20, type=int)
parser.add_argument("--max_n", default=20000, type=int)
parser.add_argument("--min_n", default=10000, type=int)
parser.add_argument("--restart_p", default=0.05, type=float)
parser.add_argument("--label_p", default=0.1, type=float)

# training hyperparameters
parser.add_argument('--meta_lr', type=float, default=0.001, help='meta learning rate')
parser.add_argument('--batch_train_n', type=int, default=10, help='shot number')
parser.add_argument('--model', type=str, default='gcn', help='gcn/gat/graphsage')
parser.add_argument('--train_strategy', type=str, default='meta', help='meta/finetune/none/nn')
parser.add_argument('--metatrain_iterations', type=int, default=3000, help='meta training iterations')
parser.add_argument('--update_batch_size', type=int, default=100, help='how much samples used for training')
parser.add_argument('--inner_train_steps', type=int, default=5, help='inner_train_step')
parser.add_argument('--inner_lr', type=float, default=1e-3, help='inner learning rate')
parser.add_argument('--inner_lr_test', type=float, default=1e-3, help='inner learning rate of test')
parser.add_argument("--test_load_epoch", default=100, type=int)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--train', type=int, default=1, help='train or test')

# model hyperparameters
parser.add_argument('--graph_threshold', type=float, default=-1, help='threshold of graph construction')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--nclasses', type=int, default=4, help='number of classes')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--proto', type=str, default='graph', help='mean or graph')
parser.add_argument('--use_proto', type=int, default=0, help='use protonet or not')
parser.add_argument('--use_maml', type=int, default=0, help='use maml or not')
parser.add_argument('--use_structure', type=int, default=0, help='use structure or not')
parser.add_argument('--use_ae', type=int, default=0, help='use autoencoder or not')
parser.add_argument('--ae_weight', type=float, default=1.0, help='the weight of autoencoder loss')
parser.add_argument('--structure_dim', type=int, default=8, help='structure dimension')
parser.add_argument('--nhops', type=int, default=2, help='number of hops of structure embedding')
parser.add_argument('--hop_concat_type', type=str, default='fc', help='fc or attention or mean')
parser.add_argument('--module_type', type=str, default='sigmoid', help='sigmoid or film')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--weight_metric', type=int, default=1, help='1: k-hop common neighbor num, 2: jacard index, 3: adar index, 4: pagerank')

args = parser.parse_args()
print(args)

assert torch.cuda.is_available()
device = torch.device('cuda:{}'.format(args.device))
torch.backends.cudnn.benchmark = True

random.seed(args.random_seed)
np.random.seed(args.random_seed)

flag = 1

exp_string = "metalr.{}_".format(args.meta_lr) + "model.{}_".format(args.model) + "ts.{}_".format(
    args.train_strategy) + "ubs.{}_".format(args.update_batch_size) + "innerlr.{}_".format(
    args.inner_lr) + "hidden.{}_".format(args.hidden) + "weightmetric.{}_".format(args.weight_metric)

if args.train_strategy == 'meta':
    exp_string += "proto.{}_".format(args.use_proto) + "maml.{}_".format(args.use_maml)
    if args.use_proto == 1:
        exp_string += "prototype.{}_".format(args.proto)
        if args.use_structure == 1:
            exp_string += "structure.{}_".format(args.use_structure) + "nhops.{}_".format(args.nhops)
            exp_string += "hopconcat.{}_".format(args.hop_concat_type) + "module.{}_".format(
                args.module_type) + "sdim.{}_".format(args.structure_dim)
        if args.use_ae == 1:
            exp_string += "ae.{}_".format(args.use_ae) + "aeweight.{}_".format(args.ae_weight)

if args.batch_train_n != 10:
    exp_string+="nshot.{}_".format(args.batch_train_n)

if flag == 0:
    args.train_strategy = 'nn'

SAVE_EPOCH = 20


def gradient_step(model, optimiser, loss_fn, x, y, **kwargs):
    """Takes a single gradient step.
    """
    model.train()
    optimiser.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss, y_pred


def train(args, model, optimiser, proto_model=None, structure_model=None, metatrain_iterations=2000,
          data_generator=None,
          verbose=True, fit_function=gradient_step, fit_function_kwargs={}):
    if verbose:
        print('Begin training...')

    for epoch in range(metatrain_iterations):

        input_data = data_generator.next_batch()

        loss, acc, _ = fit_function(args, model, optimiser, proto_model, structure_model, input_data,
                                    **fit_function_kwargs)
        print(epoch, loss.item(), acc.item())

        if not os.path.exists(args.logdir + '/' + exp_string + '/'):
            os.makedirs(args.logdir + '/' + exp_string + '/')

        if epoch % SAVE_EPOCH == 0 and epoch != 0:
            torch.save(model.state_dict(), args.logdir + '/' + exp_string + '/' + 'model_epoch_{}'.format(epoch))
            if proto_model != None:
                torch.save(proto_model.state_dict(),
                           args.logdir + '/' + exp_string + '/' + 'proto_model_epoch_{}'.format(epoch))
            if structure_model != None:
                torch.save(structure_model.state_dict(),
                           args.logdir + '/' + exp_string + '/' + 'structure_model_epoch_{}'.format(epoch))
            if args.train_strategy == 'matchingnet':
                torch.save(fit_function_kwargs['matching_net'].state_dict(),
                           args.logdir + '/' + exp_string + '/' + 'matchingnet_epoch_{}'.format(epoch))

    if verbose:
        print('Finished.')


def evaluate(args, model, optimiser, proto_model=None, structure_model=None, data_generator=None,
             verbose=True, fit_function=gradient_step, fit_function_kwargs={}):
    if verbose:
        print('Begin evaluating...')

    input_data = data_generator.test_batch()
    loss, acc, ci = fit_function(args, model, optimiser, proto_model, structure_model, input_data,
                                 **fit_function_kwargs)
    # ipdb.set_trace()
    print("testing results: loss is {}, acc is {}, ci is {}".format(loss.item(), acc.item(), ci.item()))

    if verbose:
        print('Finished.')


def main():
    if 'academic' in args.datapath:
        data_generator = DataGenerator(args)
    else:
        assert ('please enter a valid data path')
    meta_model = GCN(nfeat=args.in_f_d,
                     nhid=args.hidden,
                     nclass=args.nclasses,
                     dropout=args.dropout).to(device)
    proto_model = GCN_Proto(args, nfeat=args.hidden, dropout=args.dropout).to(device)
    if args.use_structure == 1:
        structure_model = GCN_Structure(args, nfeat=args.hidden, nhid=args.structure_dim, dropout=args.dropout).to(
            device)
    else:
        structure_model = None
    if args.train:
        # adj, features, labels, idx_train = data_generator.next_batch()
        if args.train_strategy == 'meta':
            if args.use_structure == 1:
                meta_optimiser = torch.optim.Adam(
                    list(meta_model.parameters()) + list(proto_model.parameters()) + list(structure_model.parameters()),
                    lr=args.meta_lr, weight_decay=args.weight_decay)
            else:
                meta_optimiser = torch.optim.Adam(
                    list(meta_model.parameters()) + list(proto_model.parameters()),
                    lr=args.meta_lr, weight_decay=args.weight_decay)
            train(args, meta_model, meta_optimiser, proto_model, structure_model,
                  metatrain_iterations=args.metatrain_iterations,
                  data_generator=data_generator, fit_function=meta_gradient_step,
                  fit_function_kwargs={'train': True, 'inner_train_steps': args.inner_train_steps,
                                       'inner_lr': args.inner_lr, 'batch_n': args.batch_n, 'device': device})
        else:
            print('please enter a correct train strategy: meta, finetune, nn, matchingnet')

    else:
        if args.train_strategy == 'meta':
            if args.test_load_epoch > 0:
                meta_model.load_state_dict(
                    torch.load(args.logdir + '/' + exp_string + '/' + 'model_epoch_{}'.format(args.test_load_epoch)))
                proto_model.load_state_dict(
                    torch.load(
                        args.logdir + '/' + exp_string + '/' + 'proto_model_epoch_{}'.format(args.test_load_epoch)))
                if args.use_structure == 1:
                    structure_model.load_state_dict(
                        torch.load(
                            args.logdir + '/' + exp_string + '/' + 'structure_model_epoch_{}'.format(
                                args.test_load_epoch)))
            meta_optimiser = torch.optim.Adam(list(meta_model.parameters()) + list(proto_model.parameters()),
                                              lr=args.meta_lr, weight_decay=args.weight_decay)
            evaluate(args, meta_model, meta_optimiser, proto_model, structure_model, data_generator=data_generator,
                     fit_function=meta_gradient_step,
                     fit_function_kwargs={'train': False, 'inner_train_steps': args.inner_train_steps,
                                          'inner_lr': args.inner_lr_test, 'batch_n': args.test_sample_g_n,
                                          'device': device})


if __name__ == '__main__':
    main()
