import torch
from torch.utils.tensorboard import SummaryWriter
import torch
import torch_geometric as pyg
import torchmetrics
import tqdm

from pairing.data import PairData
from pairing.data import Dataset
from pairing.data import loader
import pairing.data

import scipy
import scipy.stats

import uuid

torch.manual_seed(42)

auroc = torchmetrics.classification.MultilabelAUROC(Dataset.num_classes())

train = Dataset(is_train=True)
test = Dataset(is_train=False)
print(f"Training datapoints = {len(train)}. Test datapoints = {len(test)}.")

device="cpu"

# Construct models

dropout = .1

def make_sequential(num_layers,input_dim,output_dim,dropout):
    layers = []
    layers.append(torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim), torch.nn.ReLU(),torch.nn.Dropout(p=dropout)))
    while len(layers) < num_layers:
        layers.append(torch.nn.Sequential(torch.nn.Linear(output_dim, output_dim), torch.nn.ReLU(),torch.nn.Dropout(p=dropout)))
    return torch.nn.Sequential(*layers)

class GCN(torch.nn.Module):
    def __init__(self,num_gcn,num_linear,embedding_size):
        super(GCN, self).__init__()

        self.layers = []
        self.task = "graph"
        self.layers.append(self.build_conv_model(num_linear,Dataset.num_features(), embedding_size))
        while len(self.layers) < num_gcn:
            self.layers.append(self.build_conv_model(num_linear,embedding_size, embedding_size))

        for layer in self.layers:
          layer.to(device)

        self.post_mp = make_sequential(num_linear,embedding_size,embedding_size,dropout)
        self.post_mp.to(device)

    def build_conv_model(self, num_linear, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg.nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg.nn.GINConv(torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                  torch.nn.ReLU(), torch.nn.Dropout(p=dropout)))

    def forward(self, x, edge_index, batch_index):
        for layer in self.layers:
            x = layer(x,edge_index)

        pooled = pyg.nn.pool.global_mean_pool(x,batch_index)
        return self.post_mp(pooled)

class MixturePredictor(torch.nn.Module):
    def __init__(self,num_gcn,num_linear,embedding_size):
        super(MixturePredictor,self).__init__()

        self.gcn = GCN(num_gcn,num_linear,embedding_size)
        # Not using a sigmoid layer, because we will use BCEWithLogitsLoss which does
        # sigmoid automatically
        self.out = make_sequential(num_linear,2*embedding_size,Dataset.num_classes(),dropout)#torch.nn.Linear(2*embedding_size,Dataset.num_classes())

    def forward(self, x_s, edge_index_s, x_s_batch, x_t, edge_index_t, x_t_batch, y, *args, **kwargs):
        emb_s = self.gcn(x_s,edge_index_s,x_s_batch)
        emb_t = self.gcn(x_t,edge_index_t,x_t_batch)

        embedding = torch.cat([emb_s,emb_t],dim=1)
        return self.out(embedding)

def generate_params():
    # Hyperparameters for optimization trials
    distributions = {
        'STEPS': scipy.stats.loguniform(1e1, 1e2), 
        'LR': scipy.stats.loguniform(1e-4, 1e-1),
        'DIM': scipy.stats.randint(6,12),
        "LINEAR": scipy.stats.randint(1, 5),
        "GCN": scipy.stats.randint(1, 5),
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

    model = MixturePredictor(num_gcn=params["GCN"],num_linear=params["LINEAR"],embedding_size=2**params["DIM"]).to(device)

    bsz = int((2**18)/(2**params["DIM"]))
    print(f"BSZ={bsz}")
    train_loader = loader(train,batch_size=bsz)
    test_loader = loader(test,batch_size=bsz)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["LR"])

    def do_train_epoch():
        model.train()
        losses = []
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            
            pred = model(**batch.to_dict())
            
            loss = loss_fn(pred,batch.y)
            loss.backward()
            losses.append(loss*len(batch.y))

            optimizer.step()

        return torch.stack(losses).sum() / len(train)

    def collate_test():
        model.eval()
        preds = []
        ys = []
        for batch in test_loader:
            batch.to(device)
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

# do_train({"GCN":2,"LINEAR":2,"STEPS":2,"LR":1e-3,"DIM":2})
for _ in range(30):
    do_train(generate_params())