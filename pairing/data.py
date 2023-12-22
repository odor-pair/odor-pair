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
notes_fname = "notes.csv"

CUTOFF = 1e3

def order_pair(sm1, sm2):
    if sm1 < sm2:
        return (sm1, sm2)
    else:
        return (sm2, sm1)


def to_torch(smiles):
    graph = smiles2graph(smiles)
    tensor_keys = ["edge_index", 'edge_feat', 'node_feat']
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

        with open(os.path.join(raw_dir, jsfile), 'r') as f:
            all_data.append(json.load(f))

    return all_data


# Data set for predicting odor labels for single molecules
def get_singles():
    all_data = get_all_data()

    usable = [
        data for data in all_data
        if data["notes"] and data["blenders"] and data["smiles"]
    ]

    print(len(usable))
    return {data["smiles"]:set(data["notes"]) for data in usable}


def get_pairings():
    all_data = get_all_data()

    pairings = collections.defaultdict(dict)
    note_counts = collections.OrderedDict()
    all_smiles = collections.OrderedDict()


    usable = [
        data for data in all_data
        if data["notes"] and data["blenders"] and data["smiles"]
    ]
    print(
        f"Found {len(usable)} usable datapoints out of {len(all_data)} total.")

    name_to_smiles = {data["name"]: data["smiles"] for data in usable}
    valid = set(data["name"] for data in usable)

    dupes = 0
    self_edges = 0
    for data in usable:
        if data["smiles"] in all_smiles:
            dupes += 1
            continue
        all_smiles[data["smiles"]] = None

        for (other, note) in data["blenders"]:
            if not other in valid:
                continue
            if note == 'No flavor group found for these':
                continue

            if data["smiles"] == name_to_smiles[other]:
                self_edges += 1
                continue

            pair = order_pair(data["smiles"], name_to_smiles[other])
            pairings[pair][note] = ""
            if not note in note_counts:
                note_counts[note] = 0
            note_counts[note] += 1

    print(f"Found {dupes} duplicates and {self_edges} self-edges out of {len(usable)} datapoints.")
    print(f"Pairings per chemical = {len(pairings)/len(usable)}.")

    print(f"Found a total of {len(note_counts)} notes.")

    return pairings,  list(all_smiles.keys()),  note_counts, name_to_smiles


def multi_hot(notes, all_notes):
    notes = [n for n in notes.keys() if n in all_notes]
    indices = torch.tensor([all_notes.index(n) for n in notes])
    if len(indices) == 0:
        # Occurs when the notes in the pair were removed due to infrequency.
        raise AttributeError("Found no valid notes.")
    one_hots = torch.nn.functional.one_hot(indices, len(all_notes))
    return one_hots.sum(dim=0)


def to_pairdata(pairing, all_notes):
    ((sm1,sm2),notes) = pairing
    d1 = to_torch(sm1)
    d2 = to_torch(sm2)
    y = multi_hot(notes, all_notes)
    pd = PairData(x_s=d1["node_feat"].float(),
                  edge_attr_s=d1["edge_feat"].float(),
                  edge_index_s=d1["edge_index"],
                  smiles_s=sm1,
                  x_t=d2["node_feat"].float(),
                  edge_attr_t=d2["edge_feat"].float(),
                  edge_index_t=d2["edge_index"],
                  smiles_t=sm2,
                  y=y.float())
    return pd


def save(data_list, fname):
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), os.path.join(out_dir, fname))


def get_ys(data_list):
    ys = []
    for d in data_list:
        ys.append(d.y)
    return torch.stack(ys, dim=0)

def well_distributed(data_list):
    ys = get_ys(data_list)
    count_per_label = ys.sum(dim=0)
    return torch.all(count_per_label > 0)


def train_test_split(train_frac, test_frac, all_smiles):
    train_smiles, test_smiles = sklearn.model_selection.train_test_split(
    all_smiles, train_size=train_frac, test_size=test_frac)

    train_smiles, test_smiles = set(train_smiles), set(test_smiles)
    assert len(train_smiles.intersection(test_smiles)) == 0

    return train_smiles, test_smiles

def make_data_list(pairings,all_notes):
    data_list = []
    for pair in tqdm.tqdm(pairings):
        try:
            data_list.append(to_pairdata(pair, all_notes))
        except AttributeError:
            # Not printing anything because ogb will print error.
            continue
    return data_list

# Ensures both graphs in the pairings are in the smiles set.
def separate_data_list(data_list,smiles):
    return [d for d in data_list if d.smiles_s in smiles and d.smiles_t in smiles]

def partition(data_list, train_frac, test_frac, all_smiles):
    data_distributed = False
    i = 0
    while not data_distributed:
        i += 1
        train_smiles, test_smiles = train_test_split(train_frac, test_frac, all_smiles)

        train_data_list = separate_data_list(data_list,train_smiles)
        test_data_list = separate_data_list(data_list,test_smiles)

        data_distributed = well_distributed(train_data_list) & well_distributed(test_data_list)
        print(f"Partition attempt {i}.")
    return train_data_list, test_data_list

def get_all_notes():
    pairings, all_smiles, note_counts, _ = get_pairings()
    all_notes = [n for n, f in note_counts.items() if f > CUTOFF]
    print(f"Found {len(all_notes)} notes that appeared more than {CUTOFF} times.")
    return all_notes


def build(train_frac, test_frac, limit=None):
    pairings, all_smiles, note_counts, _ = get_pairings()

    all_notes = [n for n, f in note_counts.items() if f > CUTOFF]
    print(f"Found {len(all_notes)} notes that appeared more than {CUTOFF} times.")

    if limit:
        pairings = random.sample(sorted(pairings.items()), limit)
    else:
        pairings = sorted(pairings.items())

    data_list = make_data_list(pairings,all_notes)
    print(f"Found {len(all_notes)} notes that appeared more than {CUTOFF} times.")
    assert well_distributed(data_list)

    train_data_list, test_data_list = partition(data_list, train_frac, test_frac, all_smiles)    

    save(train_data_list, train_fname)
    save(test_data_list, test_fname)
    with open(os.path.join(out_dir, notes_fname),"w") as f:
        f.write(', '.join(all_notes))

    print(
        f"Built {len(train_data_list)} train and {len(test_data_list)} test datapoints."
    )
    print(
        f"Discarded {len(pairings)-(len(train_data_list)+len(test_data_list))} for train/test separation."
    )


class Dataset(InMemoryDataset):

    def __init__(self, is_train):
        super().__init__(out_dir)
        if is_train:
            self.data, self.slices = torch.load(
                os.path.join(out_dir, train_fname))
        else:
            self.data, self.slices = torch.load(
                os.path.join(out_dir, test_fname))

    @classmethod
    def num_classes(cls):
        return 33

    @classmethod
    def num_node_features(cls):
        return 9

    @classmethod
    def num_edge_features(cls):
        return 3


def loader(dataset, batch_size):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      follow_batch=['x_s', 'x_t'])


if __name__ == "__main__":
    build(train_frac=.85, test_frac=.15, limit=None)
