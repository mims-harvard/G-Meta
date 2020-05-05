import math

import torch
import torch.nn.functional as F
from torch import nn

from code.layers import GraphConvolution, Proto_GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, eval=False):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        if eval:
            return x
        else:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc3(x, adj)
            return F.log_softmax(x, dim=1)

    def functional_forward(self, x, adj, weights, eval=False):
        x = F.relu(self.gc1.functional_forward(x, adj, id=1, weights=weights))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2.functional_forward(x, adj, id=2, weights=weights)
        if eval:
            return x
        else:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc3.functional_forward(x, adj, id=3, weights=weights)
            return F.log_softmax(x, dim=1)


class GCN_Proto(nn.Module):
    def __init__(self, args, nfeat, dropout):
        super(GCN_Proto, self).__init__()

        self.gc_proto1 = Proto_GraphConvolution(args, nfeat, nfeat)
        if args.use_structure == 1:
            self.fc_gate_weight = nn.Linear(args.structure_dim, nfeat * nfeat)
            self.fc_gate_bias = nn.Linear(args.structure_dim, nfeat)
        self.nfeat = nfeat
        self.dropout = dropout
        self.args = args

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memory.size(0))
        self.memory.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, task_repr=None):
        if task_repr is not None:
            task_repr_cat = task_repr
            if self.args.module_type == 'sigmoid':
                task_gate_weight = torch.squeeze(torch.sigmoid(self.fc_gate_weight(task_repr_cat))).view(self.nfeat,
                                                                                                         self.nfeat)
                task_gate_bias = torch.squeeze(torch.sigmoid(self.fc_gate_bias(task_repr_cat))).view(1, self.nfeat)
                if self.args.vis_gate==1:
                    self.gate_weight_plot=task_gate_weight
                x = torch.tanh(self.gc_proto1(x, adj, task_gate_weight, task_gate_bias))
        else:
            x = torch.tanh(self.gc_proto1(x, adj))

        return torch.mean(x, dim=0)

class GCN_Structure(nn.Module):
    def __init__(self, args, nfeat, nhid, dropout):
        super(GCN_Structure, self).__init__()

        if args.use_ae:
            self.gc_decode_structure1 = GraphConvolution(nhid, nhid)

        if args.nhops > 1:
            self.gc_community_prob = GraphConvolution(nhid, nhid)
            self.gc_community_value = GraphConvolution(nhid, nhid)
            self.gc_structure3 = GraphConvolution(nhid, nhid)

        self.nhid = nhid
        self.nfeat = nfeat
        self.dropout = dropout
        self.args = args

        if args.hop_concat_type == 'fc':
            self.concat_weight = nn.Linear(2 * nhid, nhid)
        elif args.hop_concat_type == 'attention':
            self.concat_weight = nn.Parameter(torch.FloatTensor(nhid, 1))
            self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.nhid)
        self.concat_weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, adj_gd):
        gc_z = self.gc_decode_structure1(x, adj)
        decoder_adj = torch.sigmoid(torch.mm(gc_z, gc_z.transpose(0, 1)))
        return torch.mean((adj_gd - decoder_adj).pow(2))

    def forward_community(self, x, adj):
        gc_z = self.gc_community_value(x, adj)
        gc_s = F.softmax(self.gc_community_prob(x, adj), dim=1)
        x = F.normalize(torch.mm(gc_s.transpose(0, 1), gc_z), dim=0)
        return torch.mean(x, dim=0, keepdim=True)

    def forward_concat(self, x):
        if self.args.hop_concat_type == 'fc':
            return self.concat_weight(x)
        elif self.args.hop_concat_type == 'mean':
            return torch.mean(x, dim=0, keepdim=True)
        elif self.args.hop_concat_type == 'attention':
            att_weight = F.softmax(torch.mm(x, self.concat_weight), dim=0)
            return torch.sum(att_weight * x, dim=0, keepdim=True)
