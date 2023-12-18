import single.data
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
    for d in dataset:
        mol = rdkit.Chem.MolFromSmiles(d.smiles)
        fp = mfpgen.GetFingerprint(mol)
        preds.append(to_numpy(fp))
        ys.append(d.y.numpy())

    return np.stack(preds, axis=0), np.stack(ys, axis=0)

def get_train_pred_y():
    train = single.data.Dataset(is_train=True)
    test = single.data.Dataset(is_train=False)

    mfpgen = rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius=4,fpSize=2048)

    train_embed, train_y = collate(mfpgen,train)
    test_embed, test_y = collate(mfpgen,test)

    lgr = single.model.LogitRegression().fit(train_embed,train_y)

    tensor_pred = torch.tensor(lgr.predict(test_embed))
    tensor_y = torch.tensor(test_y)

    return tensor_pred, tensor_y

    # auroc = torchmetrics.classification.MultilabelAUROC(single.data.Dataset.num_classes())
    # print(auroc(tensor_pred,torch.tensor(tensor_y).int()))
    # analysis.auroc.make_chart(tensor_pred,tensor_y)