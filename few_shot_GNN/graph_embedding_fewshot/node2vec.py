import numpy as np
import networkx as nx
import os
from gensim.models import Word2Vec
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.externals import joblib
from itertools import combinations
import random
import torch

directed = True
p = 5.0
q = 1.0
num_walks = 1000
walk_length = 100
emb_size = 200
iteration = 5


def reset_array():
    x_train = []
    x_test = []
    y_train = []
    y_test = []


accuracy_meta_test = []
total_accuracy_meta_test = []

LABEL = {
      'Case_Based': 1, 'Genetic_Algorithms': 2, 'Neural_Networks': 3, 'Probabilistic_Methods': 4, 'Reinforcement_Learning': 5, 'Rule_Learning': 6, 'Theory': 7
}

LABEL2 = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']


def load_features(filename):
    ids, labels = [], []
    with open(filename, 'r') as f:
        line = f.readline();
        while line:
            line_split = line.split();
            ids.append(line_split[0]);
            labels.append(line_split[-1]);
            line = f.readline();
        return ids, labels


def load_graph(filename, id_list):
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            line_split = line.split()
            if line_split[0] in id_list and line_split[1] in id_list  and line_split[0] != line_split[1]:
                g.add_edge(line_split[0], line_split[1])
                g[line_split[0]][line_split[1]]['weight'] = 1
            line = f.readline()
    return g


def preprocess_transition_probs(g, directed = False, p=1, q=1):
    alias_nodes, alias_edges = {}, {};
    for node in g.nodes():
        probs = [g[node][nei]['weight'] for nei in sorted(g.neighbors(node))]
        norm_const = sum(probs)
        norm_probs = [float(prob)/norm_const for prob in probs]
        alias_nodes[node] = get_alias_nodes(norm_probs)
    if directed:
        for edge in g.edges():
            alias_edges[edge] = get_alias_edges(g, edge[0], edge[1], p, q)
    else:
        for edge in g.edges():
            alias_edges[edge] = get_alias_edges(g, edge[0], edge[1], p, q)
            alias_edges[(edge[1], edge[0])] = get_alias_edges(g, edge[1], edge[0], p, q)
    return alias_nodes, alias_edges


def get_alias_edges(g, src, dest, p=1, q=1):
    probs = [];
    for nei in sorted(g.neighbors(dest)):
        if nei==src:
            probs.append(g[dest][nei]['weight']/p)
        elif g.has_edge(nei, src):
            probs.append(g[dest][nei]['weight'])
        else:
            probs.append(g[dest][nei]['weight']/q)
    norm_probs = [float(prob)/sum(probs) for prob in probs]
    return get_alias_nodes(norm_probs)


def get_alias_nodes(probs):
    l = len(probs)
    a, b = np.zeros(l), np.zeros(l, dtype=np.int)
    small, large = [], []
    for i, prob in enumerate(probs):
        a[i] = l*prob
        if a[i]<1.0:
            small.append(i)
        else:
            large.append(i)
    while small and large:
        sma, lar = small.pop(), large.pop()
        b[sma] = lar
        a[lar]+=a[sma]-1.0
        if a[lar]<1.0:
            small.append(lar)
        else:
            large.append(lar)
    return b, a


def node2vec_walk(g, start, alias_nodes, alias_edges, walk_length=30):
    path = [start]
    while len(path)<walk_length:
        node = path[-1]
        neis = sorted(g.neighbors(node))
        if len(neis)>0:
            if len(path)==1:
                l = len(alias_nodes[node][0])
                idx = int(np.floor(np.random.rand()*l))
                if np.random.rand()<alias_nodes[node][1][idx]:
                    path.append(neis[idx])
                else:
                    path.append(neis[alias_nodes[node][0][idx]])
            else:
                prev = path[-2]
                l = len(alias_edges[(prev, node)][0])
                idx = int(np.floor(np.random.rand()*l))
                if np.random.rand()<alias_edges[(prev, node)][1][idx]:
                    path.append(neis[idx])
                else:
                    path.append(neis[alias_edges[(prev, node)][0][idx]])
        else:
            break
    return path 


edge_path = 'data/cora/cora.content'
label_path = 'data/cora/cora.cites'
model_path = './output_node2vec.model'
id_list, labels = load_features(edge_path)
g = load_graph(label_path, id_list)
for node in id_list:
    if not g.has_node(node):
        g.add_node(node)
if os.path.isfile(model_path):
    model = Word2Vec.load(model_path)
    print ('load model successfully')
else: 
    alias_nodes, alias_edges = preprocess_transition_probs(g, directed,p,q)
    walks = []
    idx_total = []
    for i in range(num_walks):
        r = np.array(range(len(id_list)))
        np.random.shuffle(r)
        for node in [id_list[j] for j in r]:
            walks.append(node2vec_walk(g, node, alias_nodes, alias_edges, walk_length))
    model = Word2Vec(walks, size=emb_size, min_count=0, sg=1, iter=iteration)
    model.save('output_node2vec.model')

y=[]
y_train = []
y_test = []
accuracy_meta_train = []
for temp in range(2708):
    y.append( LABEL[labels[temp]])
y = np.array(y)

class_label = [0, 1, 2, 3, 4, 5, 6]
combination = list(combinations(class_label, 2))
for i in range(len(combination)):
    print('Cross_Validation: ', i + 1)
    test_label = list(combination[i])
    train_label = [n for n in class_label if n not in test_label]
    print('Cross_Validation {} Train_Label_List {}: '.format(i + 1, train_label))
    print('Cross_Validation {} Test_Label_List {}: '.format(i + 1, test_label))
    classifier = LogisticRegression()

    for j in range(50):
        labels_local = labels.copy()
        select_class = random.sample(train_label, 2)
        print('Cross_Validation {} ITERATION {} Train_Label: {}'.format(i + 1, j + 1, select_class))
        class1_idx = []
        class2_idx = []
        for k in range(2708):
            if (labels_local[k] == LABEL2[select_class[0]]):
                class1_idx.append(k)
                labels_local[k] = LABEL2[select_class[0]]
            elif (labels_local[k] == LABEL2[select_class[1]]):
                class2_idx.append(k)
                labels_local[k] = LABEL2[select_class[1]]
        for m in range(50):
            y_train = []
            y_test = []
            class1_train = random.sample(class1_idx, 20)
            class2_train = random.sample(class2_idx, 20)
            class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
            class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
            train_idx = class1_train + class2_train
            random.shuffle(train_idx)
            test_idx = class1_test + class2_test
            random.shuffle(test_idx)

            x_train = np.zeros(emb_size)
            x_test = np.zeros(emb_size)

            for x in range(len(train_idx)):
                x_train = np.row_stack((x_train, model[id_list[x]]))
                y_train.append(y[x])
            x_train = np.delete(x_train, [0], axis=0)


            for x in range(len(test_idx)):
                x_test = np.row_stack((x_test, model[id_list[x]]))
                y_test.append(y[x])
            x_test = np.delete(x_test, [0], axis=0)

            y_train = np.array(y_train)
            y_test = np.array(y_test)
            classifier.fit(x_train, y_train)
            predictions = classifier.predict(x_test)
            acc_test = list(predictions - y_test).count(0) / len(y_test)
            accuracy_meta_train.append(acc_test)
    joblib.dump(classifier, 'classifier.model')
    print('Cross_Validation: {} Meta-Train_Accuracy: {}'.format(i + 1, torch.tensor(accuracy_meta_train).numpy().mean()))
    accuracy_meta_train = []
    labels_local = labels.copy()
    select_class = random.sample(test_label, 2)
    class1_idx = []
    class2_idx = []
    reset_array()
    print('Cross_Validation {} Test_Label {}: '.format(i + 1, select_class))

    for k in range(2708):
        if (labels_local[k] == LABEL2[select_class[0]]):
            class1_idx.append(k)
            labels_local[k] = 0
        elif (labels_local[k] == LABEL2[select_class[1]]):
            class2_idx.append(k)
            labels_local[k] = 1

    for m in range(50):
        y_train = []
        y_test = []
        class1_train = random.sample(class1_idx, 3)
        class2_train = random.sample(class2_idx, 3)
        class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
        class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
        train_idx = class1_train + class2_train
        random.shuffle(train_idx)
        test_idx = class1_test + class2_test
        random.shuffle(test_idx)

        x_train = np.zeros(emb_size)
        x_test = np.zeros(emb_size)

        for x in range(len(train_idx)):
            x_train = np.row_stack((x_train, model[id_list[x]]))
            y_train.append(y[x])
        x_train = np.delete(x_train, [0], axis=0)

        for x in range(len(test_idx)):
            x_test = np.row_stack((x_test, model[id_list[x]]))
            y_test.append(y[x])
        x_test = np.delete(x_test, [0], axis=0)

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        classifier_metaed = joblib.load('classifier.model')
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        acc_test = list(predictions-y_test).count(0)/len(y_test)
        accuracy_meta_test.append(acc_test)
        total_accuracy_meta_test.append(acc_test)
        reset_array()
    with open('cora.txt', 'a') as f:
        f.write('Cross_Validation: {} Meta-Test_Accuracy: {}'.format(i + 1, torch.tensor(
            accuracy_meta_test).numpy().mean()))
        f.write('\n')
    accuracy_meta_test = []
with open('cora.txt', 'a') as f:
    f.write('Total_Meta-Test_Accuracy: {}'.format(torch.tensor(total_accuracy_meta_test).numpy().mean()))
    f.write('\n')
    f.write('\n\n\n')