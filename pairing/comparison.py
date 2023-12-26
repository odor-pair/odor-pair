from main import MixturePredictor, GCN
from pairing.data import PairData, Dataset, loader
import analysis.fingerprint
import analysis.auroc
import analysis.best
import tqdm
import numpy as np

def collate(mfpgen,dataset):
    preds = []
    ys = []
    for i,d in enumerate(tqdm.tqdm(dataset)):
        # Caching these converts from O(n^2)
        # to O(n)
        fp1 = analysis.fingerprint.smiles_to_embed(mfpgen,d.smiles_s)
        
        fp2 = analysis.fingerprint.smiles_to_embed(mfpgen,d.smiles_t)
        
        preds.append(np.concatenate([fp1,fp2]))
        
        ys.append(d.y.numpy())

    return np.stack(preds, axis=0), np.stack(ys, axis=0)

def get_datasets():
    train = Dataset(is_train=True)
    test = Dataset(is_train=False)

    mfpgen = analysis.fingerprint.make_mfpgen()

    train_embed, train_y = collate(mfpgen,train)
    test_embed, test_y = collate(mfpgen,test)

    return train_embed, train_y, test_embed, test_y


def optimize():
    train_embed, train_y, test_embed, test_y = get_datasets()
    best_count, best_score = analysis.fingerprint.optimize_sample(train_embed, train_y, test_embed, test_y)
    print(f"Best count={best_count} w/ score={best_score}")
    make_chart(count=best_count)


def make_chart(count=None):
    train_embed, train_y, test_embed, test_y = get_datasets()
    train_embed, train_y = analysis.fingerprint.make_sample(train_embed, train_y,count)

    pred1, y1 = analysis.best.collate_test()
    pred2, y2 = analysis.fingerprint.get_test_pred_y(train_embed, train_y, test_embed, test_y)
    analysis.auroc.make_dual_chart("AUROC Comparison on Blended Pair Task by Model and Odor Label",pred1,y1,"Our Model",pred2,y2,"Molecular Fingerprints")


if __name__ == "__main__":
    make_chart()
    