import torch
from torch.utils.tensorboard import SummaryWriter
import torch
import torch_geometric as pyg
import torchmetrics
import tqdm

from pairing.data import PairData
import pairing.data

import scipy
import scipy.stats

import uuid

torch.manual_seed(42)

auroc = torchmetrics.classification.MultilabelAUROC(pairing.data.Dataset.num_classes())

train = pairing.data.Dataset(is_train=True)
test = pairing.data.Dataset(is_train=False)
print(f"Training datapoints = {len(train)}. Test datapoints = {len(test)}.")


class GCN(torch.nn.Module):
    def __init__(self,num_layers,embedding_size):
        super(GCN, self).__init__()

        self.layers = []
        self.task = "graph"
        self.dropout = .1
        self.layers.append(self.build_conv_model(pairing.data.Dataset.num_features(), embedding_size))
        while len(self.layers) < num_layers:
            self.layers.append(self.build_conv_model(embedding_size, embedding_size))

        self.post_mp = torch.nn.Sequential(torch.nn.Linear(embedding_size, embedding_size), torch.nn.ReLU(),torch.nn.Dropout(p=self.dropout))

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg.nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg.nn.GINConv(torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                  torch.nn.ReLU(), torch.nn.Dropout(p=self.dropout)))

    def forward(self, x, edge_index, batch_index):
        for layer in self.layers:
            x = layer(x,edge_index)

        pooled = pyg.nn.pool.global_mean_pool(x,batch_index)
        return self.post_mp(pooled)

class MixturePredictor(torch.nn.Module):
    def __init__(self,num_layers,embedding_size):
        super(MixturePredictor,self).__init__()

        self.gcn = GCN(num_layers,embedding_size)
        # Not using a sigmoid layer, because we will use BCEWithLogitsLoss which does
        # sigmoid automatically
        self.out = torch.nn.Linear(2*embedding_size,pairing.data.Dataset.num_classes())

    def forward(self, x_s, edge_index_s, x_s_batch, x_t, edge_index_t, x_t_batch, y, *args, **kwargs):
        emb_s = self.gcn(x_s,edge_index_s,x_s_batch)
        emb_t = self.gcn(x_t,edge_index_t,x_t_batch)

        embedding = torch.cat([emb_s,emb_t],dim=1)
        return self.out(embedding)

def generate_params():
    # Hyperparameters for optimization trials
    distributions = {
        'STEPS': scipy.stats.loguniform(1e2, 1e3), 
        'LR': scipy.stats.loguniform(1e-4, 1e-1),
        'DIM': scipy.stats.randint(6,12),
        "LAYERS": scipy.stats.randint(1, 5),
    }
    params = dict()
    for key, val in distributions.items():
        try:
            params[key] = val.rvs(1).item()
        except:
            params[key] = val.rvs(1)
    return params

def do_train(params):
    print(params)

    model = MixturePredictor(num_layers=params["LAYERS"],embedding_size=2**params["DIM"])

    bsz = 2**8
    train_loader = pairing.data.loader(train,batch_size=bsz)
    test_loader = pairing.data.loader(test,batch_size=bsz)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["LR"])

    def do_train_epoch():
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            
            pred = model(**batch.to_dict())
            
            loss = loss_fn(pred,batch.y)
            loss.backward()
            losses.append(loss*len(batch.y))

            optimizer.step()

        return torch.stack(losses).sum() / len(train)

    def collate_test():
        preds = []
        ys = []
        for batch in test_loader:
            with torch.no_grad():        
                pred = model(**batch.to_dict())

            preds.append(pred)
            ys.append(batch.y)

        return torch.cat(preds,dim=0), torch.cat(ys,dim=0)

    def get_test_loss():
        pred, y = collate_test()
        return loss_fn(pred,y)

    def get_auroc():
        pred, y = collate_test()
        return auroc(pred,y.int())


    run_name = str(uuid.uuid1())[:8]
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    best_loss = float('inf')
    for i in tqdm.tqdm(range(int(params["STEPS"]))):
        loss = do_train_epoch()
        tl = get_test_loss()
        if tl < best_loss:
            best_loss = loss
        else:
            print(f"Stopping early after {i}")
            break
        writer.add_scalars('Loss',{'train':loss,'test': tl},i)

    torch.save(model,f"{log_dir}/model.pt")
    metrics = {"auroc":get_auroc(),"completed":i}
    print(run_name,params,metrics)
    writer.add_hparams(params,metrics)
    writer.close()

do_train({"LAYERS":1,"STEPS":2,"LR":1e-3,"DIM":2})
# for _ in range(30):
#     do_train(generate_params())