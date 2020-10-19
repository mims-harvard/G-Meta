import os
import networkx as nx 
import numpy as np
import pandas as pd
import torch
import pickle
import dgl
from tqdm import tqdm
import json

# this is an example of disjoint label multiple graphs.

path = 'PATH'

# assume you have a list of DGL graphs stored in the variable dgl_Gs
dgl_Gs = [G1, G2, ...]
# assume you have an array of features where [feat_1, feat_2, ...] and each feat_i corresponding to the graph i.
feature_map = [feat1, feat2, ...]
# assume you have an array of labels where [label_1, label_2, ...] and each label_i corresponding to the graph i.
label_map = [label1, label2, ...]
# number of unique labels, e.g. 30
num_of_labels = 30 
# number of labels for each label set, ideally << num_of_labels so that each task can from different permutation of labels
num_label_set = 5

info = {}

for idx, G in enumerate(dgl_Gs):    
    # G is a dgl graph
    for j in range(len(label_map[idx])):
        info[str(idx) + '_' + str(j)] = label_map[idx][j]
            
df = pd.DataFrame.from_dict(info, orient='index').reset_index().rename(columns={"index": "name", 0: "label"})

labels = np.unique(list(range(num_of_labels)))

test_labels = np.random.choice(labels, num_label_set, False)
labels_left = [i for i in labels if i not in test_labels]
val_labels = np.random.choice(labels_left, num_label_set, False)
train_labels = [i for i in labels_left if i not in val_labels]

df[df.label.isin(train_labels)].reset_index(drop = True).to_csv(path + '/train.csv')
df[df.label.isin(val_labels)].reset_index(drop = True).to_csv(path + '/val.csv')
df[df.label.isin(test_labels)].reset_index(drop = True).to_csv(path + '/test.csv')

with open(path + '/graph_dgl.pkl', 'wb') as f:
    pickle.dump(dgl_Gs, f)
    
with open(path + '/label.pkl', 'wb') as f:
    pickle.dump(info, f)
    
np.save(path + '/features.npy', np.array(feature_map))


# for shared labels, multiple graph setting, similarly, assume you have process the following variables:

# assume you have a list of DGL graphs stored in the variable dgl_Gs
dgl_Gs = [G1, G2, ...]
# assume you have an array of features where [feat_1, feat_2, ...] and each feat_i corresponding to the graph i.
feature_map = [feat1, feat2, ...]
# assume you have an array of labels where [label_1, label_2, ...] and each label_i corresponding to the graph i.
label_map = [label1, label2, ...]
# number of unique labels, e.g. 5
num_of_labels = 5

info = {}
for idx, G in enumerate(dgl_Gs):    
    for i in tqdm(list(G.nodes)):
        info[str(idx) + '_' + str(i)] = labels_set[idx][i]

np.save(path + '/features.npy', np.array(feature_map))

with open(path + '/graph_dgl.pkl', 'wb') as f:
    pickle.dump(dgl_Gs, f)
    
with open(path + '/label.pkl', 'wb') as f:
    pickle.dump(info, f)

df = pd.DataFrame.from_dict(info, orient='index').reset_index().rename(columns={"index": "name", 0: "label"})

# for example, specify the graph idx to be used for val, test set, other graphs are put in the meta-train
folds = [[0, 23], [1, 22], [2, 21], [3, 20], [4, 19]]

for fold_n, i in enumerate(folds):
    temp_path = path + '/fold' + str(fold_n+1)
    train_graphs = list(range(len(dgl_Gs)))
    train_graphs.remove(i[0])
    train_graphs.remove(i[1])
    val_graph = i[0]
    test_graph = i[1]

    val_df = df[df.name.str.contains(str(val_graph)+'_')]
    test_df = df[df.name.str.contains(str(test_graph)+'_')]

    train_df = df[~df.index.isin(val_df.index)]
    train_df = train_df[~train_df.index.isin(test_df.index)]
    train_df.reset_index(drop = True).to_csv(temp_path + '/train.csv')
    val_df.reset_index(drop = True).to_csv(temp_path + '/val.csv')
    test_df.reset_index(drop = True).to_csv(temp_path + '/test.csv')