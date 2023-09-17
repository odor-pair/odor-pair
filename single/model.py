from main import MixturePredictor, GCN
import single.data
import pairing.data
import analysis.best
import torch_geometric as pyg
import torch
import numpy as np
import sklearn
import sklearn.linear_model
import scipy
import torchmetrics
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

train = single.data.Dataset(is_train=True)
test = single.data.Dataset(is_train=False)

model = analysis.best.model.gcn
model.eval()

def collate(dataset):
    loader = pyg.loader.DataLoader(dataset,batch_size=32)
    preds = []
    ys = []
    for batch in loader:
        with torch.no_grad():
            pred = model(**batch.to_dict())

        preds.append(pred)
        y = batch.y.reshape((len(batch),-1))
        ys.append(y)

    return torch.cat(preds, dim=0).numpy(), torch.cat(ys, dim=0).numpy()


train_x, train_y = collate(train)
test_x, test_y = collate(test)

lgr = LogitRegression().fit(train_x,train_y)

test_pred = torch.tensor(lgr.predict(test_x))
test_y = torch.tensor(test_y)

auroc = torchmetrics.classification.MultilabelAUROC(single.data.Dataset.num_classes())
print(auroc(test_pred,torch.tensor(test_y).int()))
analysis.auroc.make_chart(test_pred,test_y)




