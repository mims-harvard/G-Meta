import os
import networkx as nx 
import numpy as np
import pandas as pd
import torch
import pickle
import dgl
from tqdm import tqdm
import json

path = 'PATH'
# G1, ..., G5, each is an DGL graph.

with open(path + '/graph_dgl.pkl', 'wb') as f:
    pickle.dump([G1, G2, G3, G4, G5], f)

labels = # 
label_num = 
idx = 0
info = {}

def label_extract(G, i, info):
    info[str(idx) + '_' + str(i)] = labels[i]

for i in tqdm(G.nodes().numpy().tolist()):
    label_extract(G, i, info)

with open(path + '/label.pkl', 'wb') as f:
    pickle.dump(info, f)

df = pd.DataFrame.from_dict(info, orient='index').reset_index().rename(columns={"index": "name", 0: "label"})
df.to_csv(path + '/data.csv')

for i in range(1,6):

    labels = np.unique(list(info.values()))
    test_labels = np.random.choice(labels, label_num, False)
    labels_left = [i for i in labels if i not in test_labels]
    val_labels = np.random.choice(labels_left, label_num, False)
    train_labels = [i for i in labels_left if i not in val_labels]

    temp_path = path + '/fold'+str(i)
    df[df.label.isin(train_labels)].reset_index(drop = True).to_csv(temp_path + '/train.csv')
    df[df.label.isin(val_labels)].reset_index(drop = True).to_csv(temp_path + '/val.csv')
    df[df.label.isin(test_labels)].reset_index(drop = True).to_csv(temp_path + '/test.csv')

