import torch
import torch_geometric as pyg
from pairing.data import PairData, Dataset, loader
import pairing.data
from main import MixturePredictor, GCN, train,test
import numpy as np



import torch
from torch.utils.tensorboard import SummaryWriter
import torch
import torch_geometric as pyg
import torchmetrics
import tqdm

# Pickle needs these imported in this package.
from pairing.data import PairData, Dataset, loader
import pairing.data
import copy

import scipy
import scipy.stats

import uuid
import matplotlib.pyplot as plt


best_trial = "28b69c1c"

all_notes = pairing.data.get_all_notes()

ms = torch.load(f"runs/{best_trial}/model.pt",map_location=torch.device('cpu'))
# SOOOOO HACKY but we saved models directly. When loading from file the model params are not saved
# So we build a new model with the same params and then build.
model = MixturePredictor(embedding_size=843,num_linear=2,num_convs=1,aggr_steps=7,architecture="GIN")
model.load_state_dict(ms.state_dict())
test_loader = loader(test, batch_size=128)
device = "cpu"

def collate_test():
    model.eval()
    preds = []
    ys = []
    for batch in tqdm.tqdm(test_loader):
        batch.to(device)
        with torch.no_grad():
            pred = model(**batch.to_dict())

        preds.append(pred)
        ys.append(batch.y)

    return torch.cat(preds, dim=0), torch.cat(ys, dim=0)

pred, y = collate_test()
logits = torch.sigmoid(pred)

auroc = torchmetrics.classification.MultilabelAUROC(Dataset.num_classes(),average=None)
# Charting best and worst scores.
roc = torchmetrics.classification.MultilabelROC(2)
score = auroc(pred,y.int())
min_idx, min_val = np.nanargmin(score),np.nanmin(score)
max_idx, max_val = np.nanargmax(score),np.nanmax(score)
print(all_notes[min_idx],min_val)
print(all_notes[max_idx],max_val)


roc(pred[:,(min_idx,max_idx)],y[:,(min_idx,max_idx)].int())
fig_, ax_ = roc.plot()


total_auroc = torchmetrics.classification.MultilabelAUROC(Dataset.num_classes())(pred,y.int())
min_label = f"\"{all_notes[min_idx]}\" w/ auroc = {score[min_idx]}"
max_label = f"\"{all_notes[max_idx]}\" w/ auroc = {score[max_idx]}"
ax_.legend([min_label,max_label])
plt.title(f"AUROC of hardest and easiest odor labels.\nAUROC across all labels = {total_auroc}.")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()