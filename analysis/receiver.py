from main import MixturePredictor, GCN
from pairing.data import PairData, Dataset, loader
import pairing.data
import numpy as np
import torchmetrics
import analysis.best
import matplotlib.pyplot as plt

def make_chart(pred,y):
    all_notes = pairing.data.get_all_notes()

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

if __name__ == "__main__":
    pred, y = analysis.best.collate_test()
    make_chart(pred,y)
