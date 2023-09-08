from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
import pandas as pd
import torch
from torch_geometric.datasets import MoleculeNet, AQSOL
from torch.utils.tensorboard import SummaryWriter

import torch
import torch_geometric as pyg
import tqdm


# The dataset only has a single feature, which is the type of the heavy atom
# You can convert by indexing into data.atoms()
# Not super transferrable. We'd be better of using the set of 9 features from
# https://ogb.stanford.edu/docs/graphprop/
data = AQSOL(".")
test = AQSOL(".",split="test")

# molecule = Chem.MolFromSmiles(data[0]["smiles"])
# MolToImage(molecule).show()
torch.manual_seed(42)

embedding_size = 32


class GCN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()

        num_layers = 1
        self.input_dim = len(data.atoms())
        
        self.layer = pyg.nn.GCNConv(self.input_dim, embedding_size)

        self.out = torch.nn.Linear(embedding_size,1)

    def forward(self, x, edge_index, batch_index):
        x = torch.nn.functional.one_hot(x.long(), self.input_dim).float()

        x = self.layer(x,edge_index)
        x = torch.nn.functional.tanh(x)

        pooled = pyg.nn.pool.global_add_pool(x,batch_index)
        return self.out(pooled)


model = GCN()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

loader = pyg.loader.DataLoader(data,batch_size=32)

def do_train_step():
    for batch in loader:
        optimizer.zero_grad()
        
        pred = model(batch.x.float(),batch.edge_index,batch.batch)
        
        loss = loss_fn(pred.squeeze(),batch.y)
        loss.backward()

        optimizer.step()

    return loss

test_loader = pyg.loader.DataLoader(test,batch_size=32)

def get_test_loss():
    for batch in test_loader:
        with torch.no_grad():        
            pred = model(batch.x.float(),batch.edge_index,batch.batch)

        loss = loss_fn(pred.squeeze(),batch.y)

    return loss

writer = SummaryWriter()
for i in tqdm.tqdm(range(100)):
    loss = do_train_step()
    tl = get_test_loss()
    writer.add_scalars('Loss',{'train':loss,'test': tl},i)

writer.close()
# batches = data.batch_graphs(batch_size=32, drop_last=True)
# for batch in batches:
#     print(batch)
#     break

