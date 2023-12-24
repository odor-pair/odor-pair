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


if __name__ == "__main__":
    train = single.data.Dataset(is_train=True)
    test = single.data.Dataset(is_train=False)

    mfpgen = analysis.fingerprint.make_mfpgen()

    print("Loading train data")
    train_embed, train_y = collate(mfpgen,train)
    print("Loading test data")
    test_embed, test_y = collate(mfpgen,test)

    pred1,y1 = single.embedding.get_test_pred_y()
    pred2, y2 = analysis.fingerprint.get_test_pred_y(train_embed, train_y, test_embed, test_y)

    analysis.auroc.make_dual_chart(pred1,y1,"Our Model",pred2,y2,"Molecular Fingerprints")
