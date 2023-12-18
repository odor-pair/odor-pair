from main import MixturePredictor, GCN
from pairing.data import PairData, Dataset, loader
import pairing.data
import numpy as np
import torchmetrics
import analysis.best

import matplotlib.pyplot as plt

# Sorted based on first input
def make_dual_chart(pred1,y1,label1,pred2,y2,label2):
    all_notes = np.array(pairing.data.get_all_notes())
    auroc = torchmetrics.classification.MultilabelAUROC(Dataset.num_classes(),average=None)
    
    scores1 = auroc(pred1,y1.int()).numpy()
    scores2 = auroc(pred2,y2.int()).numpy()
    
    idcs = np.flip(np.argsort(scores1))
    scores1 = scores1[idcs]
    scores2 = scores2[idcs]
    all_notes = all_notes[idcs]
    idxs = [i for i in range(len(all_notes))]

    w = .4

    plt.figure(figsize=(15, 3))
    plt.bar(idxs,scores1,width=w,align='edge')
    plt.bar([i+w for i in idxs],scores2,width=w,align='edge')
    plt.legend([label1,label2])
    plt.axhline(y=0.5,color='grey',linestyle='dashed')
    plt.xticks(ticks=idxs,labels=all_notes,rotation=45)
    plt.title("AUROC comparison by Model and Odor Label")
    plt.tight_layout()

    for ticklabel in plt.gca().get_xticklabels():
        idx = ticklabel._x
        if scores1[idx] < .5 or scores1[idx] < scores2[idx]:
            ticklabel.set_color('r')
        else:
            ticklabel.set_color('black')

    plt.show()

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