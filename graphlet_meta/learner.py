import torch
import torch.nn.functional as F
import dgl.function as fn
import torch.nn as nn
from torch.nn import init
import dgl

import networkx as nx
import numpy as np
from scipy.special import comb
from itertools import combinations 
import networkx.algorithms.isomorphism as iso
from tqdm import tqdm

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# copied and editted from DGL Source 
class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = True
        
        #self.reset_parameters()

        self._activation = activation


    def forward(self, graph, feat, weight, bias):

        graph = graph.local_var()
        if self._norm:
            norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp).to(feat.device)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = torch.matmul(feat, weight)
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
        else:
            # aggregate first then mult W
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
            rst = torch.matmul(rst, weight)

        rst = rst * norm

        rst = rst + bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


class Classifier(nn.Module):
    def __init__(self, config, n_graphlets):
        super(Classifier, self).__init__()
        
        self.vars = nn.ParameterList()
        self.graph_conv = []
        self.config = config

        # create graphlet batch

        graphs = []
        for i in range(1, n_graphlets):
            graphs = graphs + generate_graphlet(i+1)

        self.graphlets = dgl.batch(graphs)

        for i, (name, param) in enumerate(self.config):
            if name is 'Linear':
                w = nn.Parameter(torch.ones(param[1], param[0]))
                # gain=1 according to cbfinn's implementation
                init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
            if name is 'GraphConv':
                # param: in_dim, hidden_dim
                w = nn.Parameter(torch.Tensor(param[0], param[1]))
                init.xavier_uniform_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.graph_conv.append(GraphConv(param[0], param[1], activation = F.relu))
            if name is 'Attention':
                # param[0] hidden size
                # param[1] attention_head_size
                # param[2] hidden_dim for classifier
                # param[3] n_ways

                w_q = nn.Parameter(torch.ones(param[1], param[0]))
                w_k = nn.Parameter(torch.ones(param[1], param[0]))    
                w_v = nn.Parameter(torch.ones(param[1], param[0]))

                w_l = nn.Parameter(torch.ones(param[3], param[2] + param[1]))

                init.kaiming_normal_(w_q)
                init.kaiming_normal_(w_k)
                init.kaiming_normal_(w_v)
                init.kaiming_normal_(w_l)

                self.vars.append(w_q)
                self.vars.append(w_k)
                self.vars.append(w_v)
                self.vars.append(w_l)

                #bias for attentions
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                #bias for classifier
                self.vars.append(nn.Parameter(torch.zeros(param[3])))


    def forward(self, g, to_fetch, vars = None):
        # For undirected graphs, in_degree is the same as
        # out_degree.

        if vars is None:
            vars = self.vars

        idx = 0 
        idx_gcn = 0

        h = g.in_degrees().view(-1, 1).float()
        h = h.to(device)

        h_graphlets = self.graphlets.in_degrees().view(-1, 1).float().to(device)

        for name, param in self.config:
            if name is 'GraphConv':
                w, b = vars[idx], vars[idx + 1]
                conv = self.graph_conv[idx_gcn]
                
                h = conv(g, h, w, b)
                h_graphlets = conv(g, h_graphlets, w, b)

                g.ndata['h'] = h
                self.graphlets.ndata['h'] = h_graphlets

                idx += 2 
                idx_gcn += 1

                if idx_gcn == len(self.graph_conv):
                    #h = dgl.mean_nodes(g, 'h')
                    num_nodes_ = g.batch_num_nodes
                    temp = [0] + num_nodes_
                    offset = torch.cumsum(torch.LongTensor(temp), dim = 0)[:-1].to(device)
                    h = h[to_fetch + offset]

                    # [# of grahlets, hidden_size]
                    h_graphlets = dgl.mean_nodes(self.graphlets, 'h')
                    
            if name is 'Linear':
                w, b = vars[idx], vars[idx + 1]
                h = F.linear(h, w, b)
                idx += 2

            if name is 'Attention':
                w_q, w_k, w_v, w_l = vars[idx], vars[idx + 1], vars[idx + 2], vars[idx + 3]
                b_q, b_k, b_v, b_l = vars[idx + 4], vars[idx + 5], vars[idx + 6], vars[idx + 7]

                Q = F.linear(h, w_q, b_q)
                V = F.linear(h, w_v, b_v)
                K = F.linear(h_graphlets, w_k, b_k)

                attention_scores = torch.matmul(Q, K)

                attention_probs = nn.Softmax(dim=-1)(attention_scores)
                context = torch.matmul(attention_probs, V)

                # classify layer, first concatenate the context vector 
                # with the hidden dim of center nodes
                h = torch.cat((context, h), 0)
                h = F.linear(h, w_l, b_l)

                idx += 2

        return h

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars