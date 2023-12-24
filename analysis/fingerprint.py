import rdkit
import rdkit.Chem.rdFingerprintGenerator
import numpy as np
import scipy
import sklearn
import sklearn.linear_model
import numpy as np
import torch

class LogitRegression(sklearn.linear_model.LinearRegression):
    EPS = 1e-5

    def __init__(self):
        super().__init__()

    def _clip01(self,arr):
        return np.asarray(arr).clip(self.EPS,1-self.EPS)

    def fit(self, x, p):
        p = self._clip01(p)
        y = scipy.special.logit(p)
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return scipy.special.expit(y)

def make_mfpgen(radius=4,fpSize=2048):
    return rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=fpSize)

def smiles_to_embed(mfpgen,smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    fp = mfpgen.GetFingerprint(mol)
    
    # https://github.com/rdkit/rdkit/discussions/3863
    return np.frombuffer(bytes(fp.ToBitString(), 'utf-8'), 'u1') - ord('0')

def get_test_pred_y(train_embed, train_y, test_embed, test_y):
    print(f"Fitting model w/ {len(train_embed)} datapoints and {len(test_embed)} test datapoints.")
    lgr = LogitRegression().fit(train_embed,train_y)

    tensor_pred = torch.tensor(lgr.predict(test_embed))
    tensor_y = torch.tensor(test_y)

    return tensor_pred, tensor_y
