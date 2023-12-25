import rdkit
import rdkit.Chem.rdFingerprintGenerator
import numpy as np
import scipy
import sklearn
import sklearn.linear_model
import numpy as np
import torch
import random
import scipy.stats
import analysis.auroc

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
    lgr = LogitRegression().fit(train_embed,train_y)

    tensor_pred = torch.tensor(lgr.predict(test_embed))
    tensor_y = torch.tensor(test_y)

    return tensor_pred, tensor_y

def make_sample(train_embed,train_y,count):
    if not count:
        return train_embed,train_y

    pairs = list(zip(train_embed,train_y))
    sample = random.sample(pairs,count)
    train_embed_sample, train_y_sample = zip(*sample)
    
    train_embed_sample = np.stack(train_embed_sample,axis=0)
    train_y_sample = np.stack(train_y_sample,axis=0)

    stats = train_y_sample.sum(axis=0)
    return train_embed_sample, train_y_sample

def do_trial(train_embed, train_y, test_embed, test_y, count):
    trial_train_embed, trial_train_y = make_sample(train_embed,train_y,count)
    pred, y = get_test_pred_y(trial_train_embed, trial_train_y, test_embed, test_y)
    return analysis.auroc.get_score(pred,y)

def optimize_sample(train_embed, train_y, test_embed, test_y, trials=100, replicas=3):
    count_distribution = scipy.stats.loguniform(1, len(train_embed))
    best_score = 0
    best_count = None

    for t in range(trials):
        count = int(count_distribution.rvs())

        scores = []
        for _ in range(replicas):
            scores.append(do_trial(train_embed, train_y, test_embed, test_y, count))

        score = np.mean(scores)
        
        print(scores)
        print(f"Trial {t}. Count={count} w/ score={score}")
        
        if score > best_score:
            best_score = score
            best_count = count
        
        print(f"Best count={best_count} w/ score={best_score}")

    return best_count, best_score
