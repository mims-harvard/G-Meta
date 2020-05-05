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

class Subgraphs(Dataset):
    """
    put nodes files as :
    root :
        |- subgraphs/*.nx includes all subgraphs for nodes
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, subgraph_list, subgraph2label, subgraph2center_node, train_edges, test_edges, n_way, spt_percentage, qry_percentage, batchsz):
        """

        :param root: root path of mini-subgraphnet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of subgraphs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy subgraphs per class
        """

        self.batchsz = batchsz  # batch of set, not batch of subgraphs
        self.n_way = 2  # there is no n_way for multi graph setting, each meta-set is a graph
        self.spt_percentage = spt_percentage  # for link prediction, we don't have the concept of shots, just percentage
        self.qry_percentage = qry_percentage  # for evaluation
        self.train_edges = train_edges 
        self.test_edges = test_edges

        #self.setsz = self.n_way * self.k_shot  # num of samples per set
        #self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        print('shuffle DB :%s, b:%d, %d-way, %d-spt_percentage, %d-query_percentage' % (
        mode, batchsz, self.n_way, spt_percentage, qry_percentage))

        # load subgraph list 

        self.subgraph2label = subgraph2label
        self.subgraph_list = subgraph_list
        self.subgraph2center_node = subgraph2center_node

        csvdata, dictGraphs = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
        self.data_label = []
        self.data_graph = []

        for i, (k, v) in enumerate(dictGraphs.items()):
            self.data_graph.append(v)

        for i, (k, v) in enumerate(csvdata.items()):
            self.data_label.append(v)  # [[subgraph1, subgraph2, ...], [subgraph111, ...]]
            #self.subgraph2label[k] = i + self.startidx  # {"subgraph_name[:9]":label}
        self.cls_num = len(self.data_label)
        self.graph_num = len(self.data_graph)

        self.create_batch(self.batchsz)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
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
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]

                if g_idx in dictGraphs.keys():
                    dictGraphs[g_idx].append(filename)
                else:
                    dictGraphs[g_idx] = [filename]

        return dictLabels, dictGraphs

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        episode here means batch, and it means how many sets we want to retain.
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # one loop generates one task
            # 1.select n_way classes randomly
            #print(self.cls_num)
            #print(self.n_way)
            
            selected_graph = np.random.choice(self.graph_num, 1, False)[0]  # select one graph
            support_x = []
            query_x = []
                
            # get the true index of this graph in the original dataset
            graph_idx = int(self.data_graph[selected_graph][0].split('_')[0])
            

            # 2. select k_shot + k_query for the selected graph
            #print(self.data_graph[selected_graph])
            #print(len(self.data_graph[selected_graph]))
            #print(len(self.data_graph))
            
            #selected_subgraphs_idx = np.random.choice(len(self.data_graph[selected_graph]), self.k_shot + self.k_query, False)
            
            #graph_size = len(self.data_graph[selected_graph])
            
            selected_subgraphs_train = self.train_edges[graph_idx]
            selected_subgraphs_test = self.test_edges[graph_idx]
            
            #num_train = int(graph_size * self.spt_percentage)
            #num_test = int(graph_size * self.qry_percentage)

            np.random.shuffle(selected_subgraphs_train)
            np.random.shuffle(selected_subgraphs_test)

            #indexDtrain = np.array(selected_subgraphs_idx[:num_train])  # idx for Dtrain
            #indexDtest = np.array(selected_subgraphs_idx[num_train:])  # idx for Dtest
            #support_x.append(
            #    np.array(self.data_graph[selected_graph])[np.array(selected_subgraphs_idx_train)].tolist())  # get all subgraphs filename for current Dtrain
            #query_x.append(np.array(self.data_graph[selected_graph])[np.array(selected_subgraphs_idx_test)].tolist())

            # shuffle the correponding relation between support set and query set
            #random.shuffle(support_x)
            #random.shuffle(query_x)

            # support_x: [setsz (k_shot+k_query * 1)] numbers of subgraphs   
            self.support_x_batch.append([selected_subgraphs_train])  # append set to current sets
            self.query_x_batch.append([selected_subgraphs_test])  # append sets to current sets

    def __getitem__(self, index):
        """
        get one task. support_x_batch[index], query_x_batch[index]

        """
        #print(self.support_x_batch[index])

        support_x = [self.subgraph_list[item]  # obtain a list of DGL subgraphs
                             for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array([self.subgraph2label[item]  
                              for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)
        support_center = np.array([self.subgraph2center_node[item] 
                             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        query_x = [self.subgraph_list[item]  # obtain a list of DGL subgraphs
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.subgraph2label[item]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)
        query_center = np.array([self.subgraph2center_node[item]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)
               
        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        
        # we don't want unique values for 

        #unique = np.unique(support_y)
        #random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        #support_y_relative = np.zeros(self.setsz)
        #query_y_relative = np.zeros(self.querysz)
        #for idx, l in enumerate(unique):
        #    support_y_relative[support_y == l] = idx
         #   query_y_relative[query_y == l] = idx

        # print('relative:', support_y_relative, query_y_relative)
        '''
        code for flatten images:
        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)
        # print(support_set_y)
        # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)
        '''
        # this is a set of subgraphs for one task.
        batched_graph_spt = dgl.batch(support_x)
        batched_graph_qry = dgl.batch(query_x)

        return batched_graph_spt, torch.LongTensor(support_y), batched_graph_qry, torch.LongTensor(query_y), torch.LongTensor(support_center), torch.LongTensor(query_center)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
        graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry = map(list, zip(*samples))

        return graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry


if __name__ == '__main__':
    # the following episode is to view one set of subgraphs via tensorboard.
    from matplotlib import pyplot as plt
    import time

    plt.ion()

    db = Subgraphs('../data/', mode='data', n_way=2, k_shot=1, k_query=15, batchsz=1000)

    db = DataLoader(db, 4, shuffle=True, num_workers=1, pin_memory=True, collate_fn = collate)

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

        print(x_spt)
        print(y_spt)
        break

    '''    

    for i, set_ in enumerate(db):
        # support_x: [k_shot*n_way]
        support_x, support_y, query_x, query_y = set_

        print(len(support_x))
        print(support_y.shape)
        print(query_x.shape)
        print(query_y.shape)

        time.sleep(5)
    '''