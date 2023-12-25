from main import MixturePredictor, GCN
import single.data
import single.embedding
from pairing.data import PairData, Dataset, loader
import analysis.fingerprint
import analysis.auroc
import analysis.best
import tqdm
import numpy as np

def collate(mfpgen,dataset):
    preds = []
    ys = []
    for d in dataset:
        fp = analysis.fingerprint.smiles_to_embed(mfpgen,d.smiles)
        preds.append(fp)
        ys.append(d.y.numpy())

    return np.stack(preds, axis=0), np.stack(ys, axis=0)


def get_datasets():
    train = single.data.Dataset(is_train=True)
    test = single.data.Dataset(is_train=False)

    mfpgen = analysis.fingerprint.make_mfpgen()

    train_embed, train_y = collate(mfpgen,train)
    test_embed, test_y = collate(mfpgen,test)

    return train_embed, train_y, test_embed, test_y


def optimize():
    train_embed, train_y, test_embed, test_y = get_datasets()
    best_count, best_score = analysis.fingerprint.optimize_sample(train_embed, train_y, test_embed, test_y, trials=500, replicas=10)
    print(f"Best count={best_count} w/ score={best_score}")
    make_chart(count=best_count)

def make_chart(count=None):
    train_embed, train_y, test_embed, test_y = get_datasets()
    train_embed, train_y = analysis.fingerprint.make_sample(train_embed, train_y,count)
    
    pred1,y1 = single.embedding.get_test_pred_y()
    pred2, y2 = analysis.fingerprint.get_test_pred_y(train_embed, train_y, test_embed, test_y)

    analysis.auroc.make_dual_chart(pred1,y1,"Our Model",pred2,y2,"Molecular Fingerprints")

if __name__ == "__main__":
    make_chart(None)
    

