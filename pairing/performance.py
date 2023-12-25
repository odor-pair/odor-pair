from pairing.data import PairData, Dataset, loader
import torchmetrics
import pairing.comparison
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def make_plot():
    counts = []
    all_scores = []

    for count in [100,300,500,1000,3000,5000,10000,30000,50000,100000,None]:
        scores = []
        replicas = 1
        for i in range(replicas):
            pred, y = pairing.comparison.get_fingerprint_test_pred_y(sample=count)
            auroc = torchmetrics.classification.MultilabelAUROC(Dataset.num_classes(),average=None)
            scores.append(np.mean(auroc(pred,y.int()).numpy()))
            print(i/replicas)
        print(scores)
        print(f"Count={count} w/ score={np.mean(scores)}")
        counts.append(count)
        all_scores.append(np.mean(scores))

    plt.plot(counts,all_scores)
    plt.show()

def do_optimization():
    dist = scipy.stats.loguniform(10, 110000)
    best_score = 0
    best_count = None

    for _ in range(100):
        count = int(dist.rvs())

        scores = []
        replicas = 3
        for i in range(replicas):
            pred, y = pairing.comparison.get_fingerprint_test_pred_y(sample=count)
            auroc = torchmetrics.classification.MultilabelAUROC(Dataset.num_classes(),average=None)
            scores.append(np.mean(auroc(pred,y.int()).numpy()))
            print(i/replicas)
        score = np.mean(scores)
        print(scores)
        print(f"Count={count} w/ score={score}")
        
        if score > best_score:
            best_score = score
            best_count = count
        
        print(f"Best count={best_count} w/ score={best_score}")

    print(f"Best count={best_count} w/ score={best_score}")




if __name__ == "__main__":
    do_optimization()
