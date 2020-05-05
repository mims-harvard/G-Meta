import os
import os
import random
import re
from collections import defaultdict
from itertools import *

import numpy as np
import scipy
import scipy.sparse as sp
import torch
from math import e
import math

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_struct_features(features):
    max_value = np.max(features, axis=0)

    # print (len(max_value))
    min_value = np.min(features, axis=0)
    mean_value = np.mean(features, axis=0)
    std_value = np.std(features, axis=0)

    # print (std_value)
    # print (mean_value)

    features = (features - min_value) / (max_value - min_value)
    # mean_value = np.mean(features, axis = 0)

    # print (mean_value)

    return features


class DataGenerator(object):
    def __init__(self, args):
        self.in_f_d = args.in_f_d
        self.batch_n = args.batch_n  # batch number
        self.sample_g_n = args.sample_g_n  # train sample graph number
        self.test_sample_g_n = args.test_sample_g_n  # test sample graph number
        self.max_n = args.max_n  # max graph size
        self.min_n = args.min_n  # min graph size
        self.label_p = args.label_p  # label percentage
        self.restart_p = args.restart_p
        self.A_n = args.A_n
        self.graphpath = args.graphpath
        self.class_n = 4
        self.batch_total_n = 20
        self.batch_train_n = args.batch_train_n
        # self.batch_evaluate_n = 5
        self.k_hop = 3
        self.weight_metric = 1  # 1: k-hop common neighbor num, 2: jacard index, 3: adar index, 4: pagerank

        a_a_list = [[] for k in range(self.A_n)]
        a_a_list_f = open(args.datapath + "a_a_collab_four_class.txt", "r")
        for line in a_a_list_f:
            line = line.strip()
            node_id = int(re.split(':', line)[0])
            neigh_list = re.split(':', line)[1]
            neigh_list_id = re.split(',', neigh_list)
            for j in range(len(neigh_list_id)):
                a_a_list[node_id].append(int(neigh_list_id[j]))
        a_a_list_f.close()
        self.a_a_list = a_a_list

        a_label = [0] * self.A_n
        a_class_f = open(args.datapath + "a_label_four_class.txt", "r")
        for line in a_class_f:
            line = line.strip()
            a_id = int(re.split(',', line)[0])
            class_id = int(re.split(',', line)[1])
            a_label[a_id] = class_id
        a_class_f.close()

        self.a_label = a_label

        a_text_f = np.zeros((self.A_n, args.in_f_d))
        a_t_e_f = open(args.datapath + "a_text_embed_four_class.txt", "r")
        for line in islice(a_t_e_f, 0, None):
            values = line.split()
            index = int(values[0])
            embeds = np.asarray(values[1:], dtype='float32')
            a_text_f[index] = embeds
        a_t_e_f.close()

        self.a_text_f = a_text_f

    # graph_adj = dict()

    def graph_sample(self):
        if not os.path.exists(self.graphpath):
            os.makedirs(self.graphpath)

        # train/test graph samples
        graph_n_list = np.random.randint(low=self.min_n, high=self.max_n, size=self.sample_g_n)
        test_graph_n_list = np.random.randint(low=self.min_n, high=self.max_n, size=self.test_sample_g_n)
        sample_g_n_list = [self.sample_g_n, self.test_sample_g_n]

        for k in range(len(sample_g_n_list)):
            # for k in range(1, 2):
            g_num = sample_g_n_list[k]

            for i in range(g_num):
                if k == 0:
                    graph_n = graph_n_list[i]
                    graph_f = open(self.graphpath + "graph_" + str(i) + ".txt", "w")
                elif k == 1:
                    graph_n = test_graph_n_list[i]
                    graph_f = open(self.graphpath + "test_graph_" + str(i) + ".txt", "w")

                node_set = []
                # random walk with restart
                # visit_list = []
                num = 0
                cur_node = np.random.randint(self.A_n)
                while (len(self.a_a_list[cur_node]) < 50):
                    cur_node = np.random.randint(self.A_n)
                node_set.append(cur_node)
                while num < graph_n:
                    # rand_p = random.random()
                    # if rand_p < self.restart_p:
                    # cur_node = np.random.randint(self.A_n)
                    # cur_node = random.choice(node_set)
                    # else:
                    next_node = random.choice(self.a_a_list[cur_node])
                    graph_f.write(str(cur_node) + " " + str(next_node) + "\n")
                    node_set.append(cur_node)
                    node_set.append(next_node)
                    cur_node = next_node
                    num += 1
                graph_f.close()

                node_set = list(set(node_set))
                # sample_label_n = int(len(node_set) * self.label_p)
                # random.shuffle(node_set)
                # sample_label_node = node_set[:sample_label_n]

                if k == 0:
                    node_label_f = open(self.graphpath + "graph_" + str(i) + "_label.txt", "w")
                elif k == 1:
                    node_label_f = open(self.graphpath + "test_graph_" + str(i) + "_label.txt", "w")

                for j in range(len(node_set)):
                    node_index = node_set[j]
                    node_label_f.write(str(node_index) + " " + str(self.a_label[node_index]) + "\n")

                node_label_f.close()

    def compute_struct_feature(self):
        train_test = [1, 0]

        # for m in range(len(train_test)):
        for m in range(1, 2):
            index = train_test[m]
            if index == 1:
                graph_n = self.sample_g_n
            elif index == 0:
                graph_n = self.test_sample_g_n

            for g_id in range(graph_n):
                if index == 1:
                    edges_unordered = np.genfromtxt("{}{}.txt".format(self.graphpath, "graph_" + str(g_id)),
                                                    dtype=np.int32)
                    id_label = np.genfromtxt("{}{}.txt".format(self.graphpath, "graph_" + str(g_id) + "_label"),
                                             dtype=np.dtype(int))
                    node_struct_feature_f = open(self.graphpath + "graph_" + str(g_id) + "_struct_feature.txt", "w")
                    idx = np.array(id_label[:, 0], dtype=np.int32)
                elif index == 0:
                    edges_unordered = np.genfromtxt("{}{}.txt".format(self.graphpath, "test_graph_" + str(g_id)),
                                                    dtype=np.int32)
                    id_label = np.genfromtxt("{}{}.txt".format(self.graphpath, "test_graph_" + str(g_id) + "_label"),
                                             dtype=np.dtype(int))
                    node_struct_feature_f = open(self.graphpath + "test_graph_" + str(g_id) + "_struct_feature.txt",
                                                 "w")
                    idx = np.array(id_label[:, 0], dtype=np.int32)

                a_a_co_list = [[] for k in range(self.A_n)]

                for i in range(len(edges_unordered)):
                    edge_temp = edges_unordered[i]
                    a_a_co_list[int(edge_temp[0])].append(int(edge_temp[1]))
                    a_a_co_list[int(edge_temp[1])].append(int(edge_temp[0]))

                a_degree = [0] * self.A_n
                a_triple = [0] * self.A_n
                a_triangle = [0] * self.A_n
                a_cc = [0] * self.A_n

                for j in range(self.A_n):
                    if len(a_a_co_list[j]):
                        triple_n = 0
                        triangle_n = 0
                        a_degree[j] = len(a_a_co_list[j])

                        for k in range(len(a_a_co_list[j])):
                            co_a_1 = a_a_co_list[j][k]
                            for l in range(k, len(a_a_co_list[j])):
                                co_a_2 = a_a_co_list[j][l]
                                triple_n += 1
                                if co_a_2 in a_a_co_list[co_a_1]:
                                    triangle_n += 1
                        a_triple[j] = triple_n
                        a_triangle[j] = triangle_n

                        a_cc[j] = float(3 * triangle_n) / triple_n

                for h in range(len(idx)):
                    node_id = idx[h]
                    node_struct_feature_f.write(str(node_id) + " " + str(a_degree[node_id]) \
                                                + " " + str(a_triple[node_id]) + " " + str(a_triangle[node_id]) \
                                                + " " + str(a_cc[node_id]) + "\n")

                node_struct_feature_f.close()

    # sys.exit()

    def next_batch(self):
        batch_adj = []
        batch_unweight_adj = []
        batch_features = []
        batch_struct_features = []
        batch_labels = []
        batch_idx_train = []
        batch_idx_evaluate = []
        batch_weight_matrix_train = []

        for i in range(self.batch_n):
            g_id = np.random.randint(self.sample_g_n)
            adj, unweight_adj, features, struct_features, labels, idx_train, idx_test, weight_matrix = self.load_data(
                g_id, 1)

            batch_adj.append(adj)
            batch_unweight_adj.append(unweight_adj)
            batch_features.append(features)
            batch_struct_features.append(struct_features)
            batch_labels.append(labels)
            batch_idx_train.append(idx_train)
            batch_idx_evaluate.append(idx_test)
            batch_weight_matrix_train.append(weight_matrix)

        return batch_adj, batch_unweight_adj, batch_features, batch_struct_features, batch_labels, batch_idx_train, batch_idx_evaluate, batch_weight_matrix_train

    def test_batch(self):
        batch_adj = []
        batch_unweight_adj = []
        batch_features = []
        batch_struct_features = []
        batch_labels = []
        batch_idx_train = []
        batch_idx_evaluate = []
        batch_weight_matrix_train = []

        # total_count = 0.0
        for i in range(self.test_sample_g_n):
            # g_id = np.random.randint(self.sample_g_n)
            adj, unweight_adj, features, struct_features, labels, idx_train, idx_test, weight_matrix = self.load_data(i,
                                                                                                                      0)

            batch_adj.append(adj)
            batch_unweight_adj.append(unweight_adj)
            batch_features.append(features)
            batch_struct_features.append(struct_features)
            batch_labels.append(labels)
            batch_idx_train.append(idx_train)
            batch_idx_evaluate.append(idx_test)
            batch_weight_matrix_train.append(weight_matrix)

        # 	total_count += len(labels)

        # print total_count / self.test_sample_g_n

        return batch_adj, batch_unweight_adj, batch_features, batch_struct_features, batch_labels, batch_idx_train, batch_idx_evaluate, batch_weight_matrix_train

    def load_data(self, g_id, train_test):
        if train_test == 1:
            id_label = np.genfromtxt("{}{}.txt".format(self.graphpath, "graph_" + str(g_id) + "_label"),
                                     dtype=np.dtype(int))
            struct_feature_f = open(self.graphpath + "graph_" + str(g_id) + "_struct_feature.txt", "r")
        elif train_test == 0:
            id_label = np.genfromtxt("{}{}.txt".format(self.graphpath, "test_graph_" + str(g_id) + "_label"),
                                     dtype=np.dtype(int))
            struct_feature_f = open(self.graphpath + "test_graph_" + str(g_id) + "_struct_feature.txt", "r")

        labels = encode_onehot(id_label[:, -1])

        idx = np.array(id_label[:, 0], dtype=np.int32)

        # text feature
        features = np.zeros((len(idx), self.in_f_d))
        for i in range(len(idx)):
            features[i] = self.a_text_f[idx[i]]
        features = sp.csr_matrix(features, dtype=np.float32)
        features = normalize_features(features)

        # structure feature
        struct_feature = np.zeros((len(idx), 4))
        line_n = 0
        for line in struct_feature_f:
            line = line.strip()
            f_temp = re.split(' ', line)[1:]
            for i in range(len(f_temp)):
                struct_feature[line_n][i] = f_temp[i]
            line_n += 1
        struct_feature_f.close()

        struct_feature = normalize_struct_features(struct_feature)

        # print len(struct_feature)

        idx_map = {j: i for i, j in enumerate(idx)}

        if train_test == 1:
            edges_unordered = np.genfromtxt("{}{}.txt".format(self.graphpath, "graph_" + str(g_id)), dtype=np.int32)
        elif train_test == 0:
            edges_unordered = np.genfromtxt("{}{}.txt".format(self.graphpath, "test_graph_" + str(g_id)),
                                            dtype=np.int32)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(id_label.shape[0], id_label.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # print (edges[0])
        # unweighted
        adj = (adj + sp.eye(adj.shape[0])) > 0
        adj = adj.astype(int)
        unweight_adj = adj
        adj = normalize_adj(adj)

        # if train_test == 1:
        # 	sample_label_n = int(len(labels) * self.label_p)
        # 	idx_train = np.random.randint(low = len(labels), size = sample_label_n)
        # elif train_test == 0:
        # 	idx_train = list(range(len(labels)))

        adj = torch.FloatTensor(np.array(adj.todense()))
        unweigh_adj = torch.LongTensor(np.array(unweight_adj.todense()))
        features = torch.FloatTensor(np.array(features.todense()))
        struct_feature = torch.FloatTensor(np.array(struct_feature))
        labels = torch.LongTensor(np.where(labels)[1])
        # labels = torch.FloatTensor(labels)
        # idx_train = torch.LongTensor(idx_train)

        graph_adj = defaultdict(list)
        for i in range(len(edges)):
            graph_adj[edges[i][0]].append(edges[i][1])
            graph_adj[edges[i][1]].append(edges[i][0])

        label_id_list = [[] for k in range(self.class_n)]
        for i in range(len(labels)):
            label_id_list[int(labels[i])].append(i)

        if train_test == 1:
            # weighted matrix for prototypical network

            # idx_all = [[0 for i in range(self.batch_total_n)] for j in range(self.class_n)]
            # for i in range(self.class_n):
            # 	idx_temp = np.random.choice(label_id_list[i], self.batch_total_n, replace=False)
            # 	idx_all[i] = idx_temp

            # idx_train = [[0 for i in range(self.batch_train_n)] for j in range(self.class_n)]
            # batch_test_n = self.batch_total_n - self.batch_train_n

            # idx_test = []
            # #idx_test = [[0 for i in range(batch_test_n)] for j in range(self.class_n)]
            # for i in range(self.class_n):
            # 	for j in range(self.batch_train_n):
            # 		idx_train[i][j] = idx_all[i][j]
            # 	for k in range(self.batch_train_n, self.batch_total_n):
            # 		#idx_test[i][k - self.batch_train_n] = idx_all[i][k]
            # 		idx_test.append(idx_all[i][k])

            idx_all = [[] for j in range(self.class_n)]
            for i in range(self.class_n):
                idx_all[i] = label_id_list[i]

            idx_train = [[0 for i in range(self.batch_train_n)] for j in range(self.class_n)]
            batch_test_n = self.batch_total_n - self.batch_train_n

            idx_test = []
            # idx_test = [[0 for i in range(batch_test_n)] for j in range(self.class_n)]
            for i in range(self.class_n):
                for j in range(self.batch_train_n):
                    idx_train[i][j] = idx_all[i][j]
                for k in range(self.batch_train_n, len(idx_all[i])):
                    # idx_test[i][k - self.batch_train_n] = idx_all[i][k]
                    idx_test.append(idx_all[i][k])

            weight_matrix_all = []
            graph_adj_sparse = sparse_matrix_transform(graph_adj)
            if self.weight_metric == 4:
                matrix_inv = pagerank(graph_adj_sparse, len(adj))

            for l in range(self.class_n):
                weight_matrix = [[0 for i in range(self.batch_train_n)] for j in range(self.batch_train_n)]
                for i in range(self.batch_train_n):
                    weight_matrix[i][i] = 1.0
                    for j in range(i + 1, self.batch_train_n):
                        src_id = idx_train[l][i]
                        end_id = idx_train[l][j]
                        if self.weight_metric == 1:
                            weight_temp = k_hop_common_neigh(graph_adj, src_id, end_id)
                        elif self.weight_metric == 2:
                            weight_temp = jaccard_index(graph_adj, src_id, end_id)
                        elif self.weight_metric == 3:
                            weight_temp = adar_index(graph_adj, src_id, end_id)
                        elif self.weight_metric == 4:
                            weight_temp = matrix_inv[src_id][end_id]
                        weight_matrix[i][j] = weight_temp
                        weight_matrix[j][i] = weight_temp
                weight_matrix_all.append(weight_matrix)

            weight_matrix_all = torch.FloatTensor(weight_matrix_all)

            # print weight_matrix_all[0]

            return adj, unweigh_adj, features, struct_feature, labels, idx_train, idx_test, weight_matrix_all

        elif train_test == 0:
            idx_all = [[] for j in range(self.class_n)]
            for i in range(self.class_n):
                idx_all[i] = label_id_list[i]

            idx_train = [[0 for i in range(self.batch_train_n)] for j in range(self.class_n)]
            batch_test_n = self.batch_total_n - self.batch_train_n

            idx_test = []
            # idx_test = [[0 for i in range(batch_test_n)] for j in range(self.class_n)]
            for i in range(self.class_n):
                for j in range(self.batch_train_n):
                    idx_train[i][j] = idx_all[i][j]
                for k in range(self.batch_train_n, len(idx_all[i])):
                    # idx_test[i][k - self.batch_train_n] = idx_all[i][k]
                    idx_test.append(idx_all[i][k])

            weight_matrix_all = []
            graph_adj_sparse = sparse_matrix_transform(graph_adj)
            if self.weight_metric == 4:
                matrix_inv = pagerank(graph_adj_sparse, len(adj))
            for l in range(self.class_n):
                weight_matrix = [[0 for i in range(self.batch_train_n)] for j in range(self.batch_train_n)]
                for i in range(self.batch_train_n):
                    weight_matrix[i][i] = 1.0
                    for j in range(i + 1, self.batch_train_n):
                        src_id = idx_train[l][i]
                        end_id = idx_train[l][j]
                        if self.weight_metric == 1:
                            weight_temp = k_hop_common_neigh(graph_adj, src_id, end_id)
                        elif self.weight_metric == 2:
                            weight_temp = jaccard_index(graph_adj, src_id, end_id)
                        elif self.weight_metric == 3:
                            weight_temp = adar_index(graph_adj, src_id, end_id)
                        elif self.weight_metric == 4:
                            weight_temp = matrix_inv[src_id][end_id]
                        # weight_temp = math.exp(- weight_temp)
                        weight_matrix[i][j] = weight_temp
                        weight_matrix[j][i] = weight_temp
                weight_matrix_all.append(weight_matrix)

            weight_matrix_all = torch.FloatTensor(weight_matrix_all)

            # print weight_matrix_all[0]

            return adj, unweigh_adj, features, struct_feature, labels, idx_train, idx_test, weight_matrix_all

    def test_graph_analysis(self):
        total_count = 0.0
        for g_id in range(100):
            id_label = np.genfromtxt("{}{}.txt".format(self.graphpath, "graph_" + str(g_id) + "_label"),
                                     dtype=np.dtype(int))
            edges_unordered = np.genfromtxt("{}{}.txt".format(self.graphpath, "graph_" + str(g_id)), dtype=np.int32)
            idx = np.array(id_label[:, 0], dtype=np.int32)
            idx_map = {j: i for i, j in enumerate(idx)}

            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
                edges_unordered.shape)

            total_count += len(edges)
        # a_a_list = [[] for k in range(len(idx_map))]
        # for i in range(len(edges)):
        # 	a_a_list[int(edges[i][0])].append(int(edges[i][1]))
        # 	a_a_list[int(edges[i][1])].append(int(edges[i][0]))

        # deg_sum = 0
        # for j in range(len(idx_map)):
        # 	deg_sum += len(a_a_list[j])

        # print str(g_id) + ": " + str((float(2 * len(edges))) / (len(idx_map) * (len(idx_map) - 1)))
        # print str(g_id) + ": " + str((float(deg_sum)) / len(idx_map))

        print
        total_count


def bfs(graph, start, end):
    # maintain a queue of paths
    queue = []
    # push the first path into the queue
    queue.append([start])
    while queue:
        # get the first path from the queue
        path = queue.pop(0)
        # print path
        # get the last node from the path
        node = path[-1]
        # print node
        # path found
        if node == end:
            return path
        # enumerate all adjacent nodes, construct a new path and push it into the queue
        for adjacent in graph.get(node, []):
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)


def k_hop_common_neigh(graph, src_id, end_id):  # k = 3
    src_neigh = []
    end_neigh = []

    node = [src_id, end_id]
    for l in range(len(node)):
        node_id = node[l]
        for i in range(len(graph[node_id])):
            neigh_id = int(graph[node_id][i])
            if l == 0:
                src_neigh.append(neigh_id)
            else:
                end_neigh.append(neigh_id)
            for j in range(len(graph[neigh_id])):
                neigh_id_2 = int(graph[neigh_id][j])
                if l == 0:
                    src_neigh.append(neigh_id_2)
                else:
                    end_neigh.append(neigh_id_2)
                for k in range(len(graph[neigh_id_2])):
                    neigh_id_3 = int(graph[neigh_id_2][k])
                    if l == 0:
                        src_neigh.append(neigh_id_3)
                    else:
                        end_neigh.append(neigh_id_3)

    intersect = list(set(src_neigh) & set(end_neigh))
    # union = list(set(src_neigh) | set(end_neigh))

    weight = 1 / (1 + math.exp(- len(intersect)))

    return weight


def jaccard_index(graph, src_id, end_id):
    src_neigh = []
    end_neigh = []
    node = [src_id, end_id]
    for l in range(len(node)):
        node_id = node[l]
        for i in range(len(graph[node_id])):
            neigh_id = int(graph[node_id][i])
            if l == 0:
                src_neigh.append(neigh_id)
            else:
                end_neigh.append(neigh_id)

    intersect_list = list(set(src_neigh) & set(end_neigh))
    union_list = list(set(src_neigh).union(end_neigh))

    weight = 0.5 + 0.5 * float(len(intersect_list)) / len(union_list)

    return weight


def adar_index(graph, src_id, end_id):
    src_neigh = []
    end_neigh = []
    node = [src_id, end_id]
    for l in range(len(node)):
        node_id = node[l]
        for i in range(len(graph[node_id])):
            neigh_id = int(graph[node_id][i])
            if l == 0:
                src_neigh.append(neigh_id)
            else:
                end_neigh.append(neigh_id)

    intersect_list = list(set(src_neigh) & set(end_neigh))

    weight = 1.0 / (math.log(len(intersect_list) + 5))

    return weight


def matrix_transform(graph, A_n):
    graph_matrix = [[0 for i in range(A_n)] for j in range(A_n)]
    for node_id in range(len(graph)):
        for i in range(len(graph[node_id])):
            neigh_id = int(graph[node_id][i])
            graph_matrix[node_id][neigh_id] = 1
            graph_matrix[neigh_id][node_id] = 1

    return graph_matrix


def sparse_matrix_transform(graph):
    row_ind = [k for k, v in graph.items() for _ in range(len(v))]
    col_ind = [i for ids in graph.values() for i in ids]

    X = sp.csr_matrix(([1] * len(row_ind), (row_ind, col_ind)))

    return X


def pagerank(graph, node_n):
    # graph_inv = np.linalg.pinv(graph, rcond=1e-10)
    identity_m = scipy.sparse.identity(node_n)
    new_graph = identity_m.todense() - 0.1 * graph.todense()
    # new_graph = scipy.sparse.csr_matrix(new_graph)
    graph_inv = np.linalg.pinv(new_graph)

    max_v = np.matrix.max(graph_inv)
    min_v = np.matrix.min(graph_inv)

    graph_inv = (graph_inv - min_v) / (max_v - min_v)
    graph_inv = graph_inv.tolist()

    # print (graph_inv[src][end])

    return graph_inv
