import dgl
from tqdm import tqdm
import networkx as nx
from itertools import combinations
import numpy as np
import random
import pickle

path = 'PATH'
adjs = np.load(path + '/graphs_adj.npy', allow_pickle = True)
# this .npy file is an array of 2D-array. [A1, A2, ..., An] where Ai is the adjacency matrix of graph i.

training_edges_fraction = 0.3
pos_test_edges = []
pos_val_edges = []
pos_train_edges = []
neg_test_edges = []
neg_train_edges = []
neg_val_edges = []

info = {}
info_spt = {}
info_qry = {}
total_subgraph = {}
center_nodes = {}

G_all_graphs = []

for idx_ in tqdm(range(len(adjs))):
    G = nx.from_numpy_array(adjs[idx_])
    
    adj_upp = np.multiply(adjs[idx_], np.triu(np.ones(adjs[idx_].shape)))
    x1, x2 = np.where(adj_upp == 1)
    edges = list(zip(x1, x2))
    
    # training edges
    sampled = np.random.choice(list(range(len(edges))), int(len(edges)*training_edges_fraction), replace = False)
    
    pos_train_edges.append([str(idx_) + '_' + str(i[0]) + '_' + str(i[1]) for i in np.array(edges)[sampled]])

    pos_test = [i for i in list(range(len(edges))) if i not in sampled]
    
    pos_test_edges.append([str(idx_) + '_' + str(i[0]) + '_' + str(i[1]) for i in np.array(edges)[pos_test]])
        
    G_sample = dgl.DGLGraph()
    G_sample.add_nodes(len(G.nodes))
    G_sample.add_edges(np.array(edges).T[0], np.array(edges).T[1])    
    num_pos = np.sum(adjs[idx_])/2
    
    sampled_frac = int(5*(sum(sum(adjs[idx_]))/len(G.nodes)))
    
    comb = []
    for i in list(range(len(G.nodes))):
        l = list(range(len(G.nodes)))
        l.remove(i)
        comb = comb + (list(zip([i] * sampled_frac, random.choices(l, k = sampled_frac))))

    random.shuffle(comb)
    comb_flipped = [(k,v) for v,k in comb]
    l = list(set(comb_flipped) & set(comb))

    neg_edges_sampled = [i for i in comb if i not in l]

    neg_edges = list(set(neg_edges_sampled) - set(edges) - set([(k,v) for (v,k) in edges]))
    
    np.random.seed(10)
    idx_neg = np.random.choice(list(range(len(neg_edges))), len(edges), replace = False)
    neg_edges = np.array(neg_edges)[idx_neg]
    
    idx_neg_train = np.random.choice(list(range(len(neg_edges))), len(sampled), replace = False)
    
    neg_train_edges.append([str(idx_) + '_' + str(i[0]) + '_' + str(i[1]) for i in np.array(neg_edges)[idx_neg_train]])

    neg_test = [i for i in list(range(len(neg_edges))) if i not in idx_neg_train]
    neg_test_edges.append([str(idx_) + '_' + str(i[0]) + '_' + str(i[1]) for i in np.array(neg_edges)[neg_test]])
    
    train_edges_pos = np.array(edges)[sampled]
    test_edges_pos = np.array(edges)[pos_test]
    
    train_edges_neg = np.array(neg_edges)[idx_neg_train]
    test_edges_neg = np.array(neg_edges)[neg_test]
    
    for i in np.array(neg_edges):
        # negative injection, following SEAL
        G_sample.add_edge(i[0],i[1])
    
    G_all_graphs.append(G_sample)
    
    for i in np.array(train_edges_pos):            
        node1 = i[0]
        node2 = i[1]
        
        info[str(idx_) + '_' + str(node1) + '_' + str(node2)] = 1
        info_spt[str(idx_) + '_' + str(node1) + '_' + str(node2)] = 1
        
    for i in np.array(test_edges_pos):            
        node1 = i[0]
        node2 = i[1]
        
        info[str(idx_) + '_' + str(node1) + '_' + str(node2)] = 1
        info_qry[str(idx_) + '_' + str(node1) + '_' + str(node2)] = 1
        
    for i in np.array(train_edges_neg):            
        node1 = i[0]
        node2 = i[1]
        
        info[str(idx_) + '_' + str(node1) + '_' + str(node2)] = 0
        info_spt[str(idx_) + '_' + str(node1) + '_' + str(node2)] = 0
        
    for i in np.array(test_edges_neg):            
        node1 = i[0]
        node2 = i[1]
        
        info[str(idx_) + '_' + str(node1) + '_' + str(node2)] = 0
        info_qry[str(idx_) + '_' + str(node1) + '_' + str(node2)] = 0
        
with open(path + '/graph_dgl.pkl', 'wb') as f:
    pickle.dump(G_all_graphs, f)

with open(path + '/label.pkl', 'wb') as f:
    pickle.dump(info, f)

# split on graphs
num_test_graphs = int(0.1 * len(G_all_graphs))

l = list(range(len(G_all_graphs)))
test_graphs_idx = np.random.choice(l, num_test_graphs, replace = False).tolist()

l = [i for i in l if i not in test_graphs_idx]
val_graphs_idx = np.random.choice(l, num_test_graphs, replace = False).tolist()

fold = [test_graphs_idx, val_graphs_idx]

df_spt = pd.DataFrame.from_dict(info_spt, orient='index').reset_index().rename(columns={"index": "name", 0: "label"})
df_qry = pd.DataFrame.from_dict(info_qry, orient='index').reset_index().rename(columns={"index": "name", 0: "label"})
df = pd.DataFrame.from_dict(info, orient='index').reset_index().rename(columns={"index": "name", 0: "label"})

i = fold

temp_path = path
train_graphs = list(range(len(G_all_graphs)))

train_graphs = [j for j in train_graphs if j not in i[0] + i[1]]
val_graph = i[1]
test_graph = i[0]

train_spt = pd.DataFrame()
val_spt = pd.DataFrame()
test_spt = pd.DataFrame()

train_qry = pd.DataFrame()
val_qry = pd.DataFrame()
test_qry = pd.DataFrame()

train = pd.DataFrame()
val = pd.DataFrame()
test = pd.DataFrame()

for graph_id in range(len(val_graph)):

    val_df = df_spt[df_spt.name.str.contains('^' + str(val_graph[graph_id])+'_')]
    test_df = df_spt[df_spt.name.str.contains('^' + str(test_graph[graph_id])+'_')]

    val_spt = val_spt.append(val_df)
    test_spt = test_spt.append(test_df)

    val_df = df_qry[df_qry.name.str.contains('^' + str(val_graph[graph_id])+'_')]
    test_df = df_qry[df_qry.name.str.contains('^' + str(test_graph[graph_id])+'_')]

    val_qry = val_qry.append(val_df)
    test_qry = test_qry.append(test_df)

    val_df = df[df.name.str.contains('^' + str(val_graph[graph_id])+'_')]
    test_df = df[df.name.str.contains('^' + str(test_graph[graph_id])+'_')]

    val = val.append(val_df)
    test = test.append(test_df)

val_spt.reset_index(drop = True).to_csv(temp_path + '/val_spt.csv')
test_spt.reset_index(drop = True).to_csv(temp_path + '/test_spt.csv')

val_qry.reset_index(drop = True).to_csv(temp_path + '/val_qry.csv')
test_qry.reset_index(drop = True).to_csv(temp_path + '/test_qry.csv')

val.reset_index(drop = True).to_csv(temp_path + '/val.csv')
test.reset_index(drop = True).to_csv(temp_path + '/test.csv')

train_df = df_spt[~df_spt.index.isin(val_spt.index)]
train_df = train_df[~train_df.index.isin(test_spt.index)]
train_df.reset_index(drop = True).to_csv(temp_path + '/train_spt.csv')

train_df = df_qry[~df_qry.index.isin(val_qry.index)]
train_df = train_df[~train_df.index.isin(test_qry.index)]
train_df.reset_index(drop = True).to_csv(temp_path + '/train_qry.csv')

train_df = df[~df.index.isin(val.index)]
train_df = train_df[~train_df.index.isin(test.index)]
train_df.reset_index(drop = True).to_csv(temp_path + '/train.csv')
