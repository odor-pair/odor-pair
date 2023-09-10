import torch
import tqdm
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from ogb.utils import smiles2graph

import os
import json
import collections


data_dir = "data"

dir_list = os.listdir(data_dir)
all_data = []

for jsfile in tqdm.tqdm(dir_list):
    with open(os.path.join(data_dir,jsfile),'r') as f:
        all_data.append(json.load(f))


usable = [data for data in all_data if data["notes"] and data["blenders"] and data["smiles"]]
name_to_smiles = {data["name"]:data["smiles"] for data in usable}
valid = set(data["name"] for data in usable)

print(f"Found {len(usable)} usable datapoints out of {len(all_data)} total.")

def order_pair(sm1,sm2):
    if sm1 < sm2:
        return (sm1,sm2)
    else:
        return (sm2,sm1)

def to_torch(smiles):
    if not isinstance(y,torch.Tensor):
        y = torch.tensor(y)

    graph = smiles2graph(smiles)
    tensor_keys = ["edge_index",'edge_feat','node_feat']
    for key in tensor_keys:
        graph[key] = torch.tensor(graph[key])
    return graph

data = all_data[0]
pairings = collections.defaultdict(set)
all_notes = set()

for data in usable:
    for (other,note) in data["blenders"]:
        if not other in valid:
            continue
        if note == 'No flavor group found for these':
            continue

        pair = order_pair(data["smiles"],name_to_smiles[other])
        pairings[pair].add(note)
        all_notes.add(note)

all_notes = list(all_notes)

def multi_hot(notes):
    indices = torch.tensor([all_notes.index(n) for n in notes])
    one_hots = torch.nn.functional.one_hot(indices,len(all_notes))
    return one_hots.sum(dim=0)

# https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html
print(f"Duplication={len(pairings)/len(usable)}")
print(f"Found a total of {len(all_notes)} notes.")
for pairing in pairings:
    ns = pairings[pairing]
    print(multi_hot(ns))
    break

