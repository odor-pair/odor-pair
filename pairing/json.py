import pairing.data
from pairing.data import PairData, Dataset, loader

import collections
import numpy as np

import matplotlib.pyplot as plt
import tqdm
import json

pairings, _, _, _ = pairing.data.get_pairings()
singles = pairing.data.get_singles()
pairings = {pair:list(labels.keys()) for pair, labels in pairings.items()}

all_notes = set()
def build_json(is_train):
    dataset = pairing.data.Dataset(is_train=is_train)
    results = []
    for d in tqdm.tqdm(dataset):
        pair = (d.smiles_s,d.smiles_t)
        pair2 = (d.smiles_t,d.smiles_s)
        if not pair in pairings and not pair2 in pairings:
            continue
        assert not (pair in pairings and pair2 in pairings)

        notes = pairings[pair] if pair in pairings else pairings[pair2]
        all_notes.update(notes)
        mol1_notes = list(singles[d.smiles_s])
        mol2_notes = list(singles[d.smiles_t])
        results.append({"mol1":d.smiles_s,"mol1_notes":mol1_notes,
            "mol2":d.smiles_t,"mol2_notes":mol2_notes,"blend_notes":notes})

    return results

def build_full_json():
    results = []
    for (mol1,mol2), notes in tqdm.tqdm(pairings.items()):
        mol1_notes = list(singles[mol1])
        mol2_notes = list(singles[mol2])
        results.append({"mol1":mol1,"mol1_notes":mol1_notes,
            "mol2":mol2,"mol2_notes":mol2_notes,"blend_notes":notes})
        all_notes.update(notes)
    return results


train_json = build_json(True)
print(len(train_json))
with open("dataset/train.json",'w') as f:
    json.dump(train_json,f)

test_json = build_json(False)
print(len(test_json))
with open("dataset/test.json",'w') as f:
    json.dump(test_json,f)

full_json = build_full_json()
print(len(full_json))
with open("dataset/full.json",'w') as f:
    json.dump(full_json,f)


print("Used Notes:",pairing.data.get_all_notes())
print("All Raw Notes:",all_notes)