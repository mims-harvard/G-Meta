# Meta-GNN
## Desccription of Meta-GNN
source_code for Meta-GNN (implement of Meta-GNN).<br>
--Meta-GNN: On Few-shot Node Classification in Graph Meta-learning

## Environment And Dependencies
PyTorch>=1.0.0<br> 
Install other dependencies: `$ pip install -r requirement.txt`

## Dataset
We provide the citation network datasets under `meta_gnn/data/`.

## Dataset Partition
We have shown the details of node partition in the paper.

## Usage
For Meta-GNN:
```
$ cd meta_gnn/
$ python sgc.py 
```

For Few-shot:<br>
(1)SGC:
```
$ cd sgc_fewshot/
$ python citation.py 
```
(2)GCN:
```
$ cd gcn_fewshot/
$ python citation.py 
```
(3)Deepwalk or Node2Vec:
```
$ cd graph_embedding_fewshot/
$ python deepwalk.py    //    $ python node2vec.py
```
(4)Graphsage
```
$ cd graphsage_fewshot/
$ python graphsage.py --dataset cora --gpu 0
```

## Reference
<br>Any comments and feedback are appreciated.