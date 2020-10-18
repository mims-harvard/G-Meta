import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import csv
import random
import pickle
from torch.utils.data import DataLoader
import dgl
import networkx as nx
import itertools

class Subgraphs(Dataset):
    def __init__(self, root, mode, subgraph2label, n_way, k_shot, k_query, batchsz, args, adjs, h):
        self.batchsz = batchsz  # batch of set, not batch of subgraphs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot support set
        self.k_query = k_query  # for query set
        self.setsz = self.n_way * self.k_shot  # num of samples per support set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.h = h # number of h hops
        self.sample_nodes = args.sample_nodes
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, %d-hops' % (
        mode, batchsz, n_way, k_shot, k_query, h))
    
        # load subgraph list if preprocessed
        self.subgraph2label = subgraph2label
        
        if args.link_pred_mode == 'True':
            self.link_pred_mode = True
        else:
            self.link_pred_mode = False
        
        if self.link_pred_mode:
            dictLabels_spt, dictGraphs_spt, dictGraphsLabels_spt = self.loadCSV(os.path.join(root, mode + '_spt.csv'))
            dictLabels_qry, dictGraphs_qry, dictGraphsLabels_qry = self.loadCSV(os.path.join(root, mode + '_qry.csv'))
            dictLabels, dictGraphs, dictGraphsLabels = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
        else:
            dictLabels, dictGraphs, dictGraphsLabels = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
        
        self.task_setup = args.task_setup

        self.G = []

        for i in adjs:
            self.G.append(i)
        
        self.subgraphs = {}
       
        if self.task_setup == 'Disjoint':
            self.data = []

            for i, (k, v) in enumerate(dictLabels.items()):
                self.data.append(v)  # [[subgraph1, subgraph2, ...], [subgraph111, ...]]
            self.cls_num = len(self.data)

            self.create_batch_disjoint(self.batchsz)
        elif self.task_setup == 'Shared':
            
            if self.link_pred_mode:
                
                self.data_graph_spt = []

                for i, (k, v) in enumerate(dictGraphs_spt.items()):
                    self.data_graph_spt.append(v)
                self.graph_num_spt = len(self.data_graph_spt)

                self.data_label_spt = [[] for i in range(self.graph_num_spt)]

                relative_idx_map_spt = dict(zip(list(dictGraphs_spt.keys()), range(len(list(dictGraphs_spt.keys())))))

                for i, (k, v) in enumerate(dictGraphsLabels_spt.items()):
                    for m, n in v.items():
                        self.data_label_spt[relative_idx_map_spt[k]].append(n)
                        
                self.cls_num_spt = len(self.data_label_spt[0])
                
                self.data_graph_qry = []

                for i, (k, v) in enumerate(dictGraphs_qry.items()):
                    self.data_graph_qry.append(v)
                self.graph_num_qry = len(self.data_graph_qry)

                self.data_label_qry = [[] for i in range(self.graph_num_qry)]

                relative_idx_map_qry = dict(zip(list(dictGraphs_qry.keys()), range(len(list(dictGraphs_qry.keys())))))

                for i, (k, v) in enumerate(dictGraphsLabels_qry.items()):
                    for m, n in v.items():
                        self.data_label_qry[relative_idx_map_qry[k]].append(n)
                        
                self.cls_num_qry = len(self.data_label_qry[0])
                
                self.create_batch_LinkPred(self.batchsz)

            else:    
                self.data_graph = []

                for i, (k, v) in enumerate(dictGraphs.items()):
                    self.data_graph.append(v)
                self.graph_num = len(self.data_graph)

                self.data_label = [[] for i in range(self.graph_num)]

                relative_idx_map = dict(zip(list(dictGraphs.keys()), range(len(list(dictGraphs.keys())))))

                for i, (k, v) in enumerate(dictGraphsLabels.items()):
                    #self.data_label[k] = []
                    for m, n in v.items():

                        self.data_label[relative_idx_map[k]].append(n)  # [(graph 1)[(label1)[subgraph1, subgraph2, ...], (label2)[subgraph111, ...]], graph2: [[subgraph1, subgraph2, ...], [subgraph111, ...]] ]
                self.cls_num = len(self.data_label[0])
                self.graph_num = len(self.data_graph)

                self.create_batch_shared(self.batchsz)


    def loadCSV(self, csvf):
        dictGraphsLabels = {}
        dictLabels = {}
        dictGraphs = {}

        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[1]
                g_idx = int(filename.split('_')[0])
                label = row[2]
                # append filename to current label

                if g_idx in dictGraphs.keys():
                    dictGraphs[g_idx].append(filename)
                else:
                    dictGraphs[g_idx] = [filename]
                    dictGraphsLabels[g_idx] = {}

                if label in dictGraphsLabels[g_idx].keys():
                    dictGraphsLabels[g_idx][label].append(filename)
                else:
                    dictGraphsLabels[g_idx][label] = [filename]

                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels, dictGraphs, dictGraphsLabels

    def create_batch_disjoint(self, batchsz):
        """
        create the entire set of batches of tasks for disjoint label setting, indepedent of # of graphs.
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            #print(self.cls_num)
            #print(self.n_way)
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                
                # 2. select k_shot + k_query for each class
                selected_subgraphs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)

                np.random.shuffle(selected_subgraphs_idx)
                indexDtrain = np.array(selected_subgraphs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_subgraphs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all subgraphs filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            # support_x: [setsz (k_shot+k_query * n_way)] numbers of subgraphs   
            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def create_batch_shared(self, batchsz):
        """
        create the entire set of batches of tasks for shared label setting, indepedent of # of graphs.
        """
        k_shot = self.k_shot
        k_query = self.k_query

        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # one loop generates one task
            # 1.select n_way classes randomly
            #print(self.cls_num)
            #print(self.n_way)
            
            selected_graph = np.random.choice(self.graph_num, 1, False)[0]  # select one graph
            data = self.data_label[selected_graph]

            selected_cls = np.array(list(range(len(data)))) # for multiple graph setting, we select cls_num * k_shot nodes
            np.random.shuffle(selected_cls)

            support_x = []
            query_x = []

            for cls in selected_cls:
                
                # 2. select k_shot + k_query for each class
                try:
                    selected_subgraphs_idx = np.random.choice(len(data[cls]), k_shot + k_query, False)
                    np.random.shuffle(selected_subgraphs_idx)
                    indexDtrain = np.array(selected_subgraphs_idx[:k_shot])  # idx for Dtrain
                    indexDtest = np.array(selected_subgraphs_idx[k_shot:])  # idx for Dtest
                    support_x.append(
                        np.array(data[cls])[indexDtrain].tolist())  # get all subgraphs filename for current Dtrain
                    query_x.append(np.array(data[cls])[indexDtest].tolist())
                except:
                    # this was not used in practice 
                    if len(data[cls]) >= k_shot:
                        selected_subgraphs_idx = np.array(range(len(data[cls])))
                        np.random.shuffle(selected_subgraphs_idx)
                        indexDtrain = np.array(selected_subgraphs_idx[:k_shot])  # idx for Dtrain
                        indexDtest = np.array(selected_subgraphs_idx[k_shot:])  # idx for Dtest
                        support_x.append(
                            np.array(data[cls])[indexDtrain].tolist())  # get all subgraphs filename for current Dtrain
                        
                        num_more = k_shot + k_query - len(data[cls])
                        count = 0

                        query_tmp = np.array(data[cls])[indexDtest].tolist()

                        while count <= num_more:
                            sub_cls = np.random.choice(selected_cls, 1)[0]
                            idx = np.random.choice(len(data[sub_cls]), 1)[0]
                            query_tmp = query_tmp + [np.array(data[sub_cls])[idx]]
                            count += 1
                        query_x.append(query_tmp)
                    else:
                        print('each class in a graph must have larger than k_shot entities in the current model')

            random.shuffle(support_x)
            random.shuffle(query_x)

            # support_x: [setsz (k_shot+k_query * 1)] numbers of subgraphs   
            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def create_batch_LinkPred(self, batchsz):
        """
        create the entire set of batches of tasks for shared label linked prediction setting, indepedent of # of graphs.
        """
        k_shot = self.k_shot
        k_query = self.k_query

        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        
        for b in range(batchsz):  # one loop generates one task

            selected_graph = np.random.choice(self.graph_num_spt, 1, False)[0]  # select one graph
            data_spt = self.data_label_spt[selected_graph]

            selected_cls_spt = np.array(list(range(len(data_spt)))) # for multiple graph setting, we select cls_num * k_shot nodes
            np.random.shuffle(selected_cls_spt)
            
            data_qry = self.data_label_qry[selected_graph]

            selected_cls_qry = np.array(list(range(len(data_qry)))) # for multiple graph setting, we select cls_num * k_shot nodes
            np.random.shuffle(selected_cls_qry)
            
            support_x = []
            query_x = []
                
            for cls in selected_cls_spt:
                
                selected_subgraphs_idx = np.random.choice(len(data_spt[cls]), k_shot, False)
                np.random.shuffle(selected_subgraphs_idx)
                support_x.append(
                    np.array(data_spt[cls])[selected_subgraphs_idx].tolist())  # get all subgraphs filename for current Dtrain
                
            for cls in selected_cls_qry:
                
                selected_subgraphs_idx = np.random.choice(len(data_qry[cls]), k_query, False)
                np.random.shuffle(selected_subgraphs_idx)
                query_x.append(np.array(data_qry[cls])[selected_subgraphs_idx].tolist())

            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets
        
    # helper to generate subgraphs on the fly.
    def generate_subgraph(self, G, i, item):
        if item in self.subgraphs:
            return self.subgraphs[item]
        else:
            # instead of calculating shortest distance, we find the following ways to get subgraphs are quicker
            if self.h == 2:
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                n_l = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
                h_hops_neighbor = torch.tensor(list(set(list(itertools.chain(*n_l)) + f_hop + [i]))).numpy()
            elif self.h == 1:
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                h_hops_neighbor = torch.tensor(list(set(f_hop + [i]))).numpy()
            elif self.h == 3:
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                n_2 = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
                n_3 = [[n.item() for n in G.in_edges(i)[0]] for i in list(itertools.chain(*n_2))]
                h_hops_neighbor = torch.tensor(list(set(list(itertools.chain(*n_2)) + list(itertools.chain(*n_3)) + f_hop + [i]))).numpy()
            if h_hops_neighbor.reshape(-1,).shape[0] > self.sample_nodes:
                h_hops_neighbor = np.random.choice(h_hops_neighbor, self.sample_nodes, replace = False)
                h_hops_neighbor = np.unique(np.append(h_hops_neighbor, [i]))
            
            sub = G.subgraph(h_hops_neighbor)         
            h_c = list(sub.parent_nid.numpy())
            dict_ = dict(zip(h_c, list(range(len(h_c)))))
            self.subgraphs[item] = (sub, dict_[i], h_c)
            
            return sub, dict_[i], h_c
    
    def generate_subgraph_link_pred(self, G, i, j, item):
        if item in self.subgraphs:
            return self.subgraphs[item]
        else:
            f_hop = [n.item() for n in G.in_edges(i)[0]]
            n_l = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
            h_hops_neighbor1 = torch.tensor(list(set([item for sublist in n_l for item in sublist] + f_hop + [i]))).numpy()
            
            f_hop = [n.item() for n in G.in_edges(j)[0]]
            n_l = [[n.item() for n in G.in_edges(j)[0]] for i in f_hop]
            h_hops_neighbor2 = torch.tensor(list(set([item for sublist in n_l for item in sublist] + f_hop + [j]))).numpy()
            
            h_hops_neighbor = np.union1d(h_hops_neighbor1, h_hops_neighbor2)
            
            if h_hops_neighbor.reshape(-1,).shape[0] > self.sample_nodes:
                h_hops_neighbor = np.random.choice(h_hops_neighbor, self.sample_nodes, replace = False)
                h_hops_neighbor = np.unique(np.append(h_hops_neighbor, [i, j]))
                
            sub = G.subgraph(h_hops_neighbor)         
            h_c = list(sub.parent_nid.numpy())
            dict_ = dict(zip(h_c, list(range(len(h_c)))))
            self.subgraphs[item] = (sub, [dict_[i], dict_[j]], h_c)
            
            return sub, [dict_[i], dict_[j]], h_c
    
    def __getitem__(self, index):
        """
        get one task. support_x_batch[index], query_x_batch[index]

        """
        #print(self.support_x_batch[index])
        if self.link_pred_mode:
            info = [self.generate_subgraph_link_pred(self.G[int(item.split('_')[0])], int(item.split('_')[1]), int(item.split('_')[2]), item)
                    for sublist in self.support_x_batch[index] for item in sublist]
        else:
            info = [self.generate_subgraph(self.G[int(item.split('_')[0])], int(item.split('_')[1]), item)
                    for sublist in self.support_x_batch[index] for item in sublist]

        support_graph_idx = [int(item.split('_')[0])  # obtain a list of DGL subgraphs
                             for sublist in self.support_x_batch[index] for item in sublist]
        
        support_x = [i for i, j, k in info]
        support_y = np.array([self.subgraph2label[item]  
                              for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)
        
        support_center = np.array([j for i, j, k in info]).astype(np.int32)
        support_node_idx = [k for i, j, k in info]

        
        if self.link_pred_mode:
            info = [self.generate_subgraph_link_pred(self.G[int(item.split('_')[0])], int(item.split('_')[1]), int(item.split('_')[2]), item)
                    for sublist in self.query_x_batch[index] for item in sublist]
        else:
            info = [self.generate_subgraph(self.G[int(item.split('_')[0])], int(item.split('_')[1]), item)
                    for sublist in self.query_x_batch[index] for item in sublist]

        query_graph_idx = [int(item.split('_')[0])  # obtain a list of DGL subgraphs
                             for sublist in self.query_x_batch[index] for item in sublist]
        
        query_x = [i for i, j, k in info]
        query_y = np.array([self.subgraph2label[item]  
                              for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)
        
        query_center = np.array([j for i, j, k in info]).astype(np.int32)
        query_node_idx = [k for i, j, k in info]

        if self.task_setup == 'Disjoint':
            unique = np.unique(support_y)
            random.shuffle(unique)
            # relative means the label ranges from 0 to n-way
            support_y_relative = np.zeros(self.setsz)
            query_y_relative = np.zeros(self.querysz)
            for idx, l in enumerate(unique):
                support_y_relative[support_y == l] = idx
                query_y_relative[query_y == l] = idx
             # this is a set of subgraphs for one task.
            batched_graph_spt = dgl.batch(support_x)
            batched_graph_qry = dgl.batch(query_x)

            return batched_graph_spt, torch.LongTensor(support_y_relative), batched_graph_qry, torch.LongTensor(query_y_relative), torch.LongTensor(support_center), torch.LongTensor(query_center), support_node_idx, query_node_idx, support_graph_idx, query_graph_idx
        elif self.task_setup == 'Shared':
       
            batched_graph_spt = dgl.batch(support_x)
            batched_graph_qry = dgl.batch(query_x)

            return batched_graph_spt, torch.LongTensor(support_y), batched_graph_qry, torch.LongTensor(query_y), torch.LongTensor(support_center), torch.LongTensor(query_center), support_node_idx, query_node_idx, support_graph_idx, query_graph_idx

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
        graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx = map(list, zip(*samples))

        return graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx
