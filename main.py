from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.utils import smiles2graph
import torch
import torch_geometric as pyg
import tqdm

import aqsol

torch.manual_seed(42)

data = aqsol.data.Dataset()

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train, test = torch.utils.data.random_split(data, [train_size, test_size])


embedding_size = 32


class GCN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()

        self.input_dim = data.num_features
        
        self.layer = pyg.nn.GCNConv(self.input_dim, embedding_size)
        self.layer2 = pyg.nn.GCNConv(embedding_size, embedding_size)

        self.out = torch.nn.Linear(2*embedding_size,1)

    def forward(self, x, edge_index, batch_index):
        x = self.layer(x,edge_index)
        x = torch.nn.functional.tanh(x)

        # x = self.layer2(x,edge_index)
        # x = torch.nn.functional.tanh(x)

        pooled = torch.cat([pyg.nn.pool.global_add_pool(x,batch_index),pyg.nn.pool.global_mean_pool(x,batch_index)], dim=1)
        return self.out(pooled)


model = GCN()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

train_loader = pyg.loader.DataLoader(train,batch_size=32)
test_loader = pyg.loader.DataLoader(test,batch_size=32)

def do_train_epoch():
    for batch in train_loader:
        optimizer.zero_grad()
        
        pred = model(batch.x.float(),batch.edge_index,batch.batch)
        
        loss = loss_fn(pred.squeeze(),batch.y)
        loss.backward()

        optimizer.step()

    return loss


def get_test_loss():
    for batch in test_loader:
        with torch.no_grad():        
            pred = model(batch.x.float(),batch.edge_index,batch.batch)

        loss = loss_fn(pred.squeeze(),batch.y)

    return loss

writer = SummaryWriter(comment="_one layer using mean+add pool")
for i in tqdm.tqdm(range(25)):
    loss = do_train_epoch()
    tl = get_test_loss()
    writer.add_scalars('Loss',{'train':loss,'test': tl},i)

writer.close()
# batches = data.batch_graphs(batch_size=32, drop_last=True)
# for batch in batches:
#     print(batch)
#     break

