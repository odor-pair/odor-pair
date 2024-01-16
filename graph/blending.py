import json
import torch
import tqdm
import torchmetrics
import graph.utils
import single.utils

import numpy as np
import analysis.fingerprint
import sklearn
import sklearn.model_selection
import warnings

with open("dataset/full.json") as f:
    full_data = json.load(f)

all_blend_notes = set()
all_single_notes = set()
for d in full_data:
    all_blend_notes.update(d["blend_notes"])
    all_single_notes.update(d["mol1_notes"])
    all_single_notes.update(d["mol2_notes"])

# Convert to list so indexing is faster.
print(f"Found {len(all_blend_notes)} notes in blends.")
all_blend_notes = list(graph.utils.canonize(all_blend_notes))
print(f"Canonized down to {len(all_blend_notes)} notes in blends.")
print()
print(f"Found {len(all_single_notes)} notes for single molecules.")
all_single_notes = list(single.utils.canonize(all_single_notes))
print(f"Canonized down to {len(all_single_notes)} notes for single molecules.")

def multi_hot(notes,canonical_list):
    notes = [n for n in notes if n in canonical_list]
    indices = torch.tensor([canonical_list.index(n) for n in notes])
    if len(indices) == 0:
        return torch.zeros(len(canonical_list))
        # Occurs when the notes in the pair were removed due to infrequency.
        # raise AttributeError("Found no valid notes.")
    one_hots = torch.nn.functional.one_hot(indices, len(canonical_list))
    return one_hots.sum(dim=0).float()

unions = []
intersects = []
model_from_blend_set = []
model_from_single_set = []
ys = []
empty = 0
selfloops = 0
for d in tqdm.tqdm(full_data):
    if d["mol1"] == d["mol2"]:
        selfloops += 1

    blnd = graph.utils.canonize(d["blend_notes"])
    if not blnd:
        empty += 1
        continue

    n1 = set(single.utils.canonize(d["mol1_notes"]))
    n2 = set(single.utils.canonize(d["mol2_notes"]))

    unions.append(multi_hot(n1.union(n2),all_blend_notes))
    intersects.append(multi_hot(n1.intersection(n2),all_blend_notes))
    model_from_blend_set.append(multi_hot(n1,all_blend_notes)+multi_hot(n2,all_blend_notes))
    model_from_single_set.append(multi_hot(n1,all_single_notes)+multi_hot(n2,all_single_notes))
    ys.append(multi_hot(blnd,all_blend_notes))

print(f"Found {empty} empty blends and {selfloops} self loops.")

unions = torch.stack(unions)
intersects = torch.stack(intersects)
model_from_blend_set = torch.stack(model_from_blend_set)
model_from_single_set = torch.stack(model_from_single_set)
ys = torch.stack(ys).int()

auroc = torchmetrics.classification.MultilabelAUROC(ys.shape[1])
warnings.filterwarnings("ignore", ".*samples in target*")
print("AUROC for predicting blended labels from unions of constituent labels:",auroc(unions,ys))
print()
print("AUROC for predicting blended labels from intersections of constituent labels:",auroc(intersects,ys))
print()

def train_model(title,model_ds):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(model_ds, ys)
    print(f"Train:{X_train.shape}->{y_train.shape}. Test:{X_test.shape}->{y_test.shape}.")
    lgr = analysis.fingerprint.LogitRegression().fit(X_train,y_train)
    test_pred = torch.from_numpy(lgr.predict(X_test))
    print(title,auroc(test_pred,y_test))
    print()

train_model("AUROC for model from blend -> blend:",model_from_blend_set)
train_model("AUROC for model from single -> blend:",model_from_single_set)


