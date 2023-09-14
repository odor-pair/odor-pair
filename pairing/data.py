import torch
import tqdm
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, download_url, Data
from ogb.utils import smiles2graph
import sklearn
import sklearn.model_selection

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
train_fname = "trainpairs.pt"
test_fname = "testpairs.pt"

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
    all_smiles = set()

    usable = [data for data in all_data if data["notes"] and data["blenders"] and data["smiles"]]
    print(f"Found {len(usable)} usable datapoints out of {len(all_data)} total.")

    name_to_smiles = {data["name"]:data["smiles"] for data in usable}
    valid = set(data["name"] for data in usable)

    dupes = 0
    for data in usable:
        if data["smiles"] in all_smiles:
            dupes += 1
            continue
        all_smiles.add(data["smiles"])

        for (other,note) in data["blenders"]:
            if not other in valid:
                continue
            if note == 'No flavor group found for these':
                continue

            pair = order_pair(data["smiles"],name_to_smiles[other])
            pairings[pair].add(note)
            all_notes.add(note)

    print(f"Found {dupes} duplicates out of {len(usable)} datapoints.")
    print(f"Pairings per chemical = {len(pairings)/len(usable)}.")
    
    all_notes = list(all_notes)
    print(f"Found a total of {len(all_notes)} notes.")

    return pairings, all_smiles, all_notes

def multi_hot(notes,all_notes):
    indices = torch.tensor([all_notes.index(n) for n in notes])
    one_hots = torch.nn.functional.one_hot(indices,len(all_notes))
    return one_hots.sum(dim=0)

def to_pairdata(sm1,sm2,notes,all_notes):
    d1 = to_torch(sm1)
    d2 = to_torch(sm2)
    y = multi_hot(notes,all_notes)
    pd = PairData(x_s=d1["node_feat"].float(), edge_attr_s=d1["edge_feat"].float(), edge_index_s=d1["edge_index"], x_t=d2["node_feat"].float(), edge_attr_t=d2["edge_feat"].float(), edge_index_t=d2["edge_index"],y=y.float())
    return pd

def build_data_list(pairings,smiles,all_notes):
    data_list = []
    for (sm1,sm2),notes in tqdm.tqdm(pairings):
        if not sm1 in smiles or not sm2 in smiles:
            continue
        try:
            pd = to_pairdata(sm1,sm2,notes,all_notes)
            data_list.append(pd)
        except AttributeError:
            # Not printing anything because ogb will print error.
            continue
    return data_list

def save(data_list,fname):
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), os.path.join(out_dir,fname))


def build(train_frac, test_frac, limit=None):
    pairings, all_smiles, all_notes = get_pairings()

    if limit:
        pairings = random.sample(sorted(pairings.items()),limit)
    else:
        pairings = sorted(pairings.items())

    train_smiles, test_smiles = sklearn.model_selection.train_test_split(list(all_smiles), train_size=train_frac, test_size=test_frac)
    train_smiles, test_smiles = set(train_smiles), set(test_smiles)
    assert len(train_smiles.intersection(test_smiles)) == 0

    train_data_list = build_data_list(pairings,train_smiles,all_notes)
    save(train_data_list,train_fname)

    test_data_list = build_data_list(pairings,test_smiles,all_notes)
    save(test_data_list,test_fname)

    print(f"Built {len(train_data_list)} train and {len(test_data_list)} test datapoints.")
    print(f"Discarded {len(pairings)-(len(train_data_list)+len(test_data_list))} for train/test separation.")

class Dataset(InMemoryDataset):
    def __init__(self,is_train):
        super().__init__("pairing")
        if is_train:
            self.data, self.slices = torch.load(os.path.join(out_dir,train_fname))
        else:
            self.data, self.slices = torch.load(os.path.join(out_dir,test_fname))

    @classmethod
    def num_classes(cls):
        return 109

    @classmethod
    def num_features(cls):
        return 9

def loader(dataset,batch_size):
    return DataLoader(dataset, batch_size=batch_size,follow_batch=['x_s', 'x_t'] )

if __name__ == "__main__":
    build(train_frac=.8,test_frac=.2)
