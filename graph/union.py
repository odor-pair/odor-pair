import json
import torch
import tqdm
import torchmetrics

import numpy as np
import analysis.fingerprint
import sklearn
import sklearn.model_selection


with open("dataset/full.json") as f:
    full_data = json.load(f)

all_notes = set()
for d in full_data:
    all_notes.update(d["blend_notes"])

dataset = []
for d in full_data:
    n1 = all_notes.intersection(d["mol1_notes"])
    n2 = all_notes.intersection(d["mol2_notes"])
    dataset.append((n1,n2,d["blend_notes"]))

all_notes = list(all_notes)

def multi_hot(notes):
    notes = [n for n in notes if n in all_notes]
    indices = torch.tensor([all_notes.index(n) for n in notes])
    if len(indices) == 0:
        return torch.zeros(len(all_notes))
        # Occurs when the notes in the pair were removed due to infrequency.
        # raise AttributeError("Found no valid notes.")
    one_hots = torch.nn.functional.one_hot(indices, len(all_notes))
    return one_hots.sum(dim=0).float()

unions = []
intersects = []
sums = []
ys = []
for (n1,n2,blnd) in tqdm.tqdm(dataset):
    unions.append(multi_hot(n1.union(n2)))
    intersects.append(multi_hot(n1.intersection(n2)))
    sums.append(multi_hot(n1)+multi_hot(n2))
    ys.append(multi_hot(blnd))

unions = torch.stack(unions)
intersects = torch.stack(intersects)
sums = torch.stack(sums)
ys = torch.stack(ys).int()

auroc = torchmetrics.classification.MultilabelAUROC(ys.shape[1])
print("AUROC for predicting blended labels from unions of constituent labels:",auroc(unions,ys))
print("AUROC for predicting blended labels from intersections of constituent labels:",auroc(intersects,ys))

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(sums, ys)

lgr = analysis.fingerprint.LogitRegression().fit(X_train,y_train)

test_pred = torch.from_numpy(lgr.predict(X_test))
print("AUROC for predicting blended labels through logistic regression:",auroc(test_pred,y_test))


