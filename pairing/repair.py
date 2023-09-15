# I accidentally built the dataset without including smiles in the DataPairs
# I didn't want to rebuild the dataset with new data because that would 
# break train/test separation and retraining models would cost too much compute
# This file repairs the dataset to include smiles.

import torch
# Pickle needs these imported in this package.
from pairing.data import PairData, Dataset, loader
import pairing.data
import collections
import tqdm
import random


def sort_t(x):
    return x[x[:, 0].sort()[1]].long()

# Because the node features are generated in arbitrary order
# we sort by row. Because tensors are hashed by id, convert to string.
def tensor_to_key(x):
    return str(sort_t(x))

def graph_to_key(x,edge_attr):
    return tensor_to_key(x) + tensor_to_key(edge_attr)

def smile_to_key(sm):
    tr = pairing.data.to_torch(sm)
    return graph_to_key(tr["node_feat"],tr["edge_feat"])

sm_to_torch = dict()
pairings, all_smiles, all_notes = pairing.data.get_pairings()
for p in tqdm.tqdm(pairings):
    # Could do some error checking to ensure there is no collision.
    if not p[0] in sm_to_torch:
        try:
            sm_to_torch[p[0]] = smile_to_key(p[0])
        except AttributeError:
            # Not printing anything because ogb will print error.
            continue

    if not p[1] in sm_to_torch:
        try:
            sm_to_torch[p[1]] = smile_to_key(p[1])
        except AttributeError:
            # Not printing anything because ogb will print error.
            continue


train = Dataset(is_train=True)
test = Dataset(is_train=False)

torch_to_sm = {v:k for k,v in sm_to_torch.items()}

def repair(dataset,fname):
    for i,d in tqdm.tqdm(enumerate(dataset),total=len(dataset)):
        key_s = graph_to_key(d.x_s,d.edge_attr_s)
        sm_s = torch_to_sm[key_s]
        d.smiles_s = sm_s
        
        key_t = graph_to_key(d.x_t,d.edge_attr_t)
        sm_t = torch_to_sm[key_t]
        d.smiles_s = sm_t

    data_list = [d for d in dataset]
    pairing.data.save(data_list,fname)

repair(train,pairing.data.train_fname)
repair(test,pairing.data.test_fname)
