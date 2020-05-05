import math

import torch
from torch import nn
from torch.autograd import Variable


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def functional_forward(self, input, adj, id, weights):
        support = torch.mm(input, weights['gc{}.weight'.format(id)])
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + weights['gc{}.bias'.format(id)]
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Proto_GraphConvolution(nn.Module):

    def __init__(self, args, in_features, out_features, bias=True):
        super(Proto_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.args = args
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.args = args

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, task_gate_weight=None, task_gate_bias=None, task_gate_weight_move=None,
                task_gate_bias_move=None):
        if task_gate_weight is not None:
            if self.args.module_type == 'sigmoid':
                support = torch.mm(input, self.weight * task_gate_weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            if task_gate_bias is not None:
                if self.args.module_type == 'sigmoid':
                    return output + self.bias * task_gate_bias
            else:
                return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'