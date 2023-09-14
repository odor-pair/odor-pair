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
import copy

import scipy
import scipy.stats

import uuid

torch.manual_seed(42)

auroc = torchmetrics.classification.MultilabelAUROC(Dataset.num_classes())

train = Dataset(is_train=True)
test = Dataset(is_train=False)
print(f"Training datapoints = {len(train)}. Test datapoints = {len(test)}.")

device = "cpu"

# Construct models

dropout = 0


def make_sequential(num_layers, input_dim, output_dim, dropout, is_last=False):
    layers = []
    layers.append(
        torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim),
                            torch.nn.ReLU(), torch.nn.Dropout(p=dropout)))
    while len(layers) < num_layers:
        layers.append(
            torch.nn.Sequential(torch.nn.Linear(output_dim, output_dim),
                                torch.nn.ReLU(), torch.nn.Dropout(p=dropout)))

    # No dropout or non-linearity
    if is_last:
        if num_layers == 1:
            layers[-1] = torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim))
        else:
            layers[-1] = torch.nn.Sequential(
                torch.nn.Linear(output_dim, output_dim))

    return torch.nn.Sequential(*layers)


class GCN(torch.nn.Module):

    def __init__(self, num_convs, num_linear, embedding_size, aggr_steps):
        super(GCN, self).__init__()

        self.layers = []
        self.task = "graph"

        # In order to tie the weights of all convolutions, the input is first
        # padded with zeros to reach embedding size.
        self.pad = torch.nn.ZeroPad2d(
            (0, embedding_size - Dataset.num_features(), 0, 0))

        self.num_convs = num_convs
        self.gcn = self.build_conv_model(num_linear, embedding_size,
                                         embedding_size)
        self.gcn.to(device)

        self.readout = pyg.nn.aggr.Set2Set(embedding_size, aggr_steps)
        self.readout.to(device)

        # Set2Set returns an output 2x wider than the input.
        # The GCN returns an output of embedding_size.
        self.post_mp = make_sequential(num_linear,
                                       2 * embedding_size,
                                       embedding_size,
                                       dropout,
                                       is_last=True)
        self.post_mp.to(device)

    def build_conv_model(self, num_linear, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg.nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg.nn.GINConv(
                torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(p=dropout)))

    def forward(self, x, edge_index, batch_index):
        x = self.pad(x)
        for _ in range(self.num_convs):
            x = self.gcn(x, edge_index)

        pooled = self.readout(x, index=batch_index)
        return self.post_mp(pooled)


class MixturePredictor(torch.nn.Module):

    def __init__(self, num_convs, num_linear, embedding_size, aggr_steps):
        super(MixturePredictor, self).__init__()

        self.gcn = GCN(num_convs, num_linear, embedding_size, aggr_steps)
        # Not using a sigmoid layer, because we will use BCEWithLogitsLoss which does
        # sigmoid automatically
        self.out = make_sequential(num_linear,
                                   2 * embedding_size,
                                   Dataset.num_classes(),
                                   dropout,
                                   is_last=True)

    def forward(self, x_s, edge_index_s, x_s_batch, x_t, edge_index_t,
                x_t_batch, y, *args, **kwargs):
        emb_s = self.gcn(x_s, edge_index_s, x_s_batch)
        emb_t = self.gcn(x_t, edge_index_t, x_t_batch)

        embedding = torch.cat([emb_s, emb_t], dim=1)
        return self.out(embedding)


def generate_params():
    # Hyperparameters for optimization trials
    distributions = {
        # Paper used 540 epochs
        # 'STEPS': scipy.stats.loguniform(1e2, 5e2),
        'STEPS': scipy.stats.randint(500, 501),
        # Paper uses [1e-5,5e-4] but we will use much larger rates
        # and stop early.
        'LR': scipy.stats.loguniform(1e-3, 1e-1),
        'DIM': scipy.stats.randint(5, 8),
        "LINEAR": scipy.stats.randint(1, 5),
        # From paper
        "CONVS": scipy.stats.randint(3, 9),
        # From paper
        "AGGR": scipy.stats.randint(1, 13),
        # From paper
        "DECAY": scipy.stats.loguniform(1e-2, 1),
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

    model = MixturePredictor(num_convs=params["CONVS"],
                             num_linear=params["LINEAR"],
                             embedding_size=2**params["DIM"],
                             aggr_steps=params["AGGR"])
    model = model.to(device)

    bsz = int((2**18) / (2**params["DIM"]))
    print(f"BSZ={bsz}")
    train_loader = loader(train, batch_size=bsz)
    test_loader = loader(test, batch_size=bsz)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["LR"])

    # From paper
    end_step = .9 * params["STEPS"]
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                  start_factor=1,
                                                  end_factor=params["DECAY"],
                                                  total_iters=end_step)

    def do_train_epoch():
        model.train()
        losses = []
        for batch in train_loader:
            print(batch.to_dict())
            batch.to(device)
            optimizer.zero_grad()

            pred = model(**batch.to_dict())

            loss = loss_fn(pred, batch.y)
            loss.backward()
            losses.append(loss * len(batch.y))

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

        return torch.cat(preds, dim=0), torch.cat(ys, dim=0)

    def get_test_loss():
        pred, y = collate_test()
        if pred.sum() == 0:
            print("HITTING ALL 0")
        return loss_fn(pred, y)

    def get_auroc():
        pred, y = collate_test()
        return auroc(pred, y.int())

    run_name = str(uuid.uuid1())[:8]
    # log_dir = f"/content/drive/MyDrive/Pygeom/runs/{run_name}"
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    best_loss = float('inf')
    patience = 0
    best = copy.deepcopy(model)
    for s in tqdm.tqdm(range(int(params["STEPS"]))):
        loss = do_train_epoch()
        scheduler.step()
        tl = get_test_loss()
        if tl < best_loss:
            best = copy.deepcopy(model)
            best_loss = loss
            patience = 0
        else:
            print(f"Stopping early after {s}")
            break
        writer.add_scalars('Loss', {'train': loss, 'test': tl}, s)

    model = best
    torch.save(model, f"{log_dir}/model.pt")
    metrics = {"auroc": get_auroc(), "completed": s}
    print(run_name, metrics, params, sep="\n")
    writer.add_hparams(params, metrics)
    writer.close()


# do_train({"CONVS":2,"LINEAR":1,"STEPS":3,"LR":2e-5,"DIM":6})
# do_train({"CONVS":1,"LINEAR":2,"STEPS":4,"LR":3e-3,"DIM":4})
for _ in range(30):
    do_train(generate_params())
