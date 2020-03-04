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

    def __init__(self, root,mode, batchsz, n_way, k_shot, k_query, path_s = 'list_subgraph.pkl', path_l = 'label.pkl'):
        """

        :param root: root path of mini-subgraphnet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of subgraphs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy subgraphs per class
        """

        self.batchsz = batchsz  # batch of set, not batch of subgraphs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query' % (
        mode, batchsz, n_way, k_shot, k_query))

        # load subgraph list 
        with open(os.path.join(root, path_s), 'rb') as f:
            subgraph_list = pickle.load(f)

        with open(os.path.join(root, path_l), 'rb') as f:
            subgraph2label = pickle.load(f)    

        self.subgraph2label = subgraph2label
        self.subgraph_list = subgraph_list

        csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
        self.data = []

        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[subgraph1, subgraph2, ...], [subgraph111, ...]]
            #self.subgraph2label[k] = i + self.startidx  # {"subgraph_name[:9]":label}
        self.cls_num = len(self.data)

        self.create_batch(self.batchsz)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[1]
                label = row[2]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        episode here means batch, and it means how many sets we want to retain.
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
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

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:

        b: number of tasks
        setsz: the size for each task, support set size (k_shot)*n_way

        :param x_spt:   [b, setsz], where each unit is a subgraph, i.e. x_spt[0] is a list of # setsz subgraphs
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz]
        :param y_qry:   [b, querysz]

        """
        #print(self.support_x_batch[index])

        support_x = [self.subgraph_list[item]  # obtain a list of DGL subgraphs
                             for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array([self.subgraph2label[item]  
                              for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)


        query_x = [self.subgraph_list[item]  # obtain a list of DGL subgraphs
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.subgraph2label[item]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

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

        return batched_graph_spt, torch.LongTensor(support_y_relative), batched_graph_qry, torch.LongTensor(query_y_relative)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
        graphs_spt, labels_spt, graph_qry, labels_qry = map(list, zip(*samples))

        return graphs_spt, labels_spt, graph_qry, labels_qry


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