from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.utils import smiles2graph
import torch
import torch_geometric as pyg
import torchmetrics
import tqdm

from pairing.data import PairData
import pairing.data

torch.manual_seed(42)

auroc = torchmetrics.classification.MultilabelAUROC(pairing.data.Dataset.num_classes())

data = pairing.data.Dataset()

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train, test = torch.utils.data.random_split(data, [train_size, test_size])


class GCN(torch.nn.Module):
    embedding_size = 32

    def __init__(self):
        super(GCN, self).__init__()

        self.input_dim = data.num_features
        
        self.layer = pyg.nn.GCNConv(self.input_dim, GCN.embedding_size)

    def forward(self, x, edge_index, batch_index):
        x = self.layer(x,edge_index)
        x = torch.nn.functional.tanh(x)

        pooled = pyg.nn.pool.global_mean_pool(x,batch_index)
        return torch.nn.functional.tanh(pooled)

class MixturePredictor(torch.nn.Module):
    def __init__(self):
        super(MixturePredictor,self).__init__()

        self.gcn = GCN()
        self.out = torch.nn.Linear(2*GCN.embedding_size,pairing.data.Dataset.num_classes())

    def forward(self, x_s, edge_index_s, x_s_batch, x_t, edge_index_t, x_t_batch, y, *args, **kwargs):
        emb_s = self.gcn(x_s,edge_index_s,x_s_batch)
        emb_t = self.gcn(x_t,edge_index_t,x_t_batch)

        embedding = torch.cat([emb_s,emb_t],dim=1)

        # Not using a sigmoid layer, because we will use BCEWithLogitsLoss which does
        # sigmoid automatically
        return self.out(embedding)


model = MixturePredictor()
train_loader = pairing.data.loader(train,batch_size=32)
test_loader = pairing.data.loader(test,batch_size=32)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

def do_train_epoch():
    for batch in train_loader:
        optimizer.zero_grad()
        
        pred = model(**batch.to_dict())
        
        loss = loss_fn(pred,batch.y)
        loss.backward()

        optimizer.step()

    return loss

def collate_test():
    preds = []
    ys = []
    for batch in test_loader:
        with torch.no_grad():        
            pred = model(**batch.to_dict())

        preds.append(pred)
        ys.append(batch.y)

    return torch.cat(preds[:2],dim=0), torch.cat(ys[:2],dim=0)

def get_test_loss():
    pred, y = collate_test()
    return loss_fn(pred,y)

def get_auroc():
    pred, y = collate_test()
    return auroc(pred,y.int())


steps = 100
writer = SummaryWriter()
for i in tqdm.tqdm(range(steps)):
    loss = do_train_epoch()
    tl = get_test_loss()
    writer.add_scalars('Loss',{'train':loss,'test': tl},i)

writer.add_hparams({"steps":steps},{"auroc":get_auroc()})
writer.close()
# batches = data.batch_graphs(batch_size=32, drop_last=True)
# for batch in batches:
#     print(batch)
#     break

