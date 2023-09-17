from main import MixturePredictor, GCN
from pairing.data import PairData, Dataset, loader
import pairing.data
import numpy as np
import torchmetrics
import analysis.best

import matplotlib.pyplot as plt

def make_chart(pred,y):
    all_notes = np.array(pairing.data.get_all_notes())
    auroc = torchmetrics.classification.MultilabelAUROC(Dataset.num_classes(),average=None)
    scores = auroc(pred,y.int()).numpy()

    idcs = np.flip(np.argsort(scores))
    scores = scores[idcs]
    all_notes = all_notes[idcs]

    plt.figure(figsize=(15, 3))
    plt.bar(all_notes,scores,align='edge')
    plt.axhline(y=0.5,color='grey',linestyle='dashed')
    plt.xticks(rotation=45)
    plt.title("AUROC by Odor Label")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pred, y = analysis.best.collate_test()
    make_chart(pred,y)