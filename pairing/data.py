import torch
import tqdm
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, download_url, Data
from ogb.utils import smiles2graph
import torchmetrics

import os
import json
import collections
import random

# The ordering of the graphs in this is arbitrary (alphabetically based on SMILES)
# but the logistic predictor does rely on this ordering.
class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'y':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


raw_dir = "data"
out_dir = "pairing"
out_name = "allpairs.pt"

def order_pair(sm1,sm2):
    if sm1 < sm2:
        return (sm1,sm2)
    else:
        return (sm2,sm1)

def to_torch(smiles):
    graph = smiles2graph(smiles)
    tensor_keys = ["edge_index",'edge_feat','node_feat']
    for key in tensor_keys:
        graph[key] = torch.tensor(graph[key])
    return graph

def get_all_data():
    dir_list = os.listdir(raw_dir)
    all_data = []

    for jsfile in tqdm.tqdm(dir_list):
        _, ftype = os.path.splitext(jsfile)
        if ftype != ".json":
            continue

        with open(os.path.join(raw_dir,jsfile),'r') as f:
            all_data.append(json.load(f))

    return all_data

def get_pairings():
    all_data = get_all_data()

    pairings = collections.defaultdict(set)
    all_notes = set()

    usable = [data for data in all_data if data["notes"] and data["blenders"] and data["smiles"]]
    print(f"Found {len(usable)} usable datapoints out of {len(all_data)} total.")

    name_to_smiles = {data["name"]:data["smiles"] for data in usable}
    valid = set(data["name"] for data in usable)

    for data in usable:
        for (other,note) in data["blenders"]:
            if not other in valid:
                continue
            if note == 'No flavor group found for these':
                continue

            pair = order_pair(data["smiles"],name_to_smiles[other])
            pairings[pair].add(note)
            all_notes.add(note)

    print(f"Duplication = {len(pairings)/len(usable)}.")
    
    all_notes = list(all_notes)
    print(f"Found a total of {len(all_notes)} notes.")

    return pairings, all_notes

def multi_hot(notes,all_notes):
    indices = torch.tensor([all_notes.index(n) for n in notes])
    one_hots = torch.nn.functional.one_hot(indices,len(all_notes))
    return one_hots.sum(dim=0)

def to_pairdata(sm1,sm2,notes,all_notes):
    d1 = to_torch(sm1)
    d2 = to_torch(sm2)
    y = multi_hot(notes,all_notes)
    pd = PairData(x_s=d1["node_feat"].float(), edge_index_s=d1["edge_index"], x_t=d2["node_feat"].float(), edge_index_t=d2["edge_index"],y=y.float())
    return pd

def build(limit=None):
    pairings, all_notes = get_pairings()

    if limit:
        pairings = random.sample(sorted(pairings.items()),limit)
    else:
        pairings = sorted(pairings.items())

    data_list = []
    for (sm1,sm2),notes in tqdm.tqdm(pairings):
        try:
            pd = to_pairdata(sm1,sm2,notes,all_notes)
            data_list.append(pd)
        except AttributeError:
            # Not printing anything because ogb will print error.
            continue

    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), os.path.join(out_dir,out_name))

class Dataset(InMemoryDataset):
    def __init__(self):
        super().__init__("pairing")
        self.data, self.slices = torch.load(os.path.join(out_dir,out_name))

    @classmethod
    def num_classes(cls):
        return 109

    @classmethod
    def num_features(cls):
        return 9

def loader(dataset,batch_size):
    return DataLoader(dataset, batch_size=batch_size,follow_batch=['x_s', 'x_t'] )

# Baseline auroc using mean of labels is 0.5
def baseline():
    data = Dataset()
    auroc = torchmetrics.classification.MultilabelAUROC(Dataset.num_classes())

    ys = []
    for d in data:
        ys.append(d.y)
    y = torch.stack(ys,dim=0)
    pred = y.mean(dim=0).unsqueeze(0)
    print(pred)
    pred = pred.expand(len(ys),-1)

    score = auroc(pred,y.int())
    print(f"Baseline auroc using mean of labels is {score}")

if __name__ == "__main__":
    # build()
    baseline()