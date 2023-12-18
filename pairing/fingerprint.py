from pairing.data import PairData, Dataset, loader
import rdkit
import rdkit.Chem.rdFingerprintGenerator
import numpy as np
import single.model
import analysis.auroc
import torch
import torchmetrics
import tqdm

# https://github.com/rdkit/rdkit/discussions/3863
def to_numpy(fp):
    return np.frombuffer(bytes(fp.ToBitString(), 'utf-8'), 'u1') - ord('0')

def collate(mfpgen,dataset):
    preds = []
    ys = []
    for i,d in enumerate(tqdm.tqdm(dataset)):
        # Caching these converts from O(n^2)
        # to O(n)
        mol1 = rdkit.Chem.MolFromSmiles(d.smiles_s)
        fp1 = mfpgen.GetFingerprint(mol1)
        
        mol2 = rdkit.Chem.MolFromSmiles(d.smiles_t)
        fp2 = mfpgen.GetFingerprint(mol2)
        
        preds.append(np.concatenate([to_numpy(fp1),to_numpy(fp2)]))
        
        ys.append(d.y.numpy())

    return np.stack(preds, axis=0), np.stack(ys, axis=0)

def get_test_pred_y():
    train = Dataset(is_train=True)
    test = Dataset(is_train=False)

    mfpgen = rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius=4,fpSize=2048)

    print("Loading train data")
    train_embed, train_y = collate(mfpgen,train)
    print("Loading test data")
    test_embed, test_y = collate(mfpgen,test)

    print("Fitting model")
    lgr = single.model.LogitRegression().fit(train_embed,train_y)

    tensor_pred = torch.tensor(lgr.predict(test_embed))
    tensor_y = torch.tensor(test_y)

    return tensor_pred, tensor_y

    # auroc = torchmetrics.classification.MultilabelAUROC(single.data.Dataset.num_classes())
    # print(auroc(tensor_pred,torch.tensor(tensor_y).int()))
    # analysis.auroc.make_chart(tensor_pred,tensor_y)

if __name__ == "__main__":
    get_test_pred_y()