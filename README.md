## G-Meta Preliminary Code and Instruction

Dependencies:
pytorch
dgl
python 3.7
numpy
networkx
scipy
tqdm
sklearn
pandas
seaborn


Training and evaluation code. Example:

```bash
# for arxiv-ogbn 0.451
python train.py --data_dir ARXIV_DATA_PATH --fold_n 1 --method ProtoMAML --task_setup Disjoint
# for tissue-PPI 0.768
python train.py --data_dir TISSUE_PPI_DATA_PATH --fold_n 1 --method ProtoMAML --task_setup Shared
# for fold-PPI 0.561
python train.py --data_dir FOLD_PPI_DATA_PATH --fold_n 1 --method ProtoMAML --task_setup Disjoint
# for FirstMM-DB 0.784
python train.py --data_dir FIRSTMMDB_DATA_PATH --fold_n 1 --method ProtoMAML --task_setup Shared --link_pred_mode True
# for Tree-of-Life 0.722
python train.py --data_dir Tree-of-life_DATA_PATH --fold_n 1 --method ProtoMAML --task_setup Shared --link_pred_mode True

```

The above script performs both training and validation and result is reported in test set. 


#### Data

Synthetic dataset: in the syn-cycle and syn-BA folder once you unzip the G-Meta_Data Folder

Real-world dataset: 

arxiv-ogbn: https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv
tissue-PPI: http://snap.stanford.edu/graphsage/ppi.zip
fold-PPI: provided in the data folder
FirstMM-DB: http://first-mm.eu/
tree-of-life:https://snap.stanford.edu/tree-of-life/data.html

A sample data preprocessing script is provided in the folder. See data_process_link.py and data_process_node.py

#### Pretrained Model

A sample pretrained model (FirstMM-DB Link Prediction) is provided in the folder as model/pt. 

More clean instruction, all the pretrained model and data processing would be open-sourced. 