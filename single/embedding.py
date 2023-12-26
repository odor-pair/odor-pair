from main import MixturePredictor, GCN
import single.data
import pairing.data
import analysis.best
import torch_geometric as pyg
import torch
import numpy as np
import torchmetrics
import analysis.fingerprint
import analysis.auroc
import single.map

def collate(model,dataset):
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


def get_test_pred_y():
    train = single.data.Dataset(is_train=True)
    test = single.data.Dataset(is_train=False)

    model = analysis.best.get_model().gcn
    model.eval()

    train_embed, train_y = collate(model,train)
    test_embed, test_y = collate(model,test)

    lgr = analysis.fingerprint.LogitRegression().fit(train_embed,train_y)

    tensor_pred = torch.tensor(lgr.predict(test_embed))
    tensor_y = torch.tensor(test_y)
    return tensor_pred, tensor_y

def do_map():
    embed = np.concatenate((train_embed,test_embed))
    y = np.concatenate((train_y,test_y))
    single.map.make_chart(embed,y)


