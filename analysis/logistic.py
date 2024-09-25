from pairing.data import PairData, Dataset, loader
import rdkit
import rdkit.Chem.rdFingerprintGenerator
import numpy as np
import single.model
import analysis.auroc
import torch
import torchmetrics
import tqdm

def get_test_pred_y(train,test):
    train_embed, train_y = collate(mfpgen,train)
    test_embed, test_y = collate(mfpgen,test)

    print(f"Fitting model w/ {len(train)} datapoints and {len(test)} test datapoints.")
    lgr = single.model.LogitRegression().fit(train_embed,train_y)

    tensor_pred = torch.tensor(lgr.predict(test_embed))
    tensor_y = torch.tensor(test_y)

    return tensor_pred, tensor_y