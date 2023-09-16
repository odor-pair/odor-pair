import torch
import torch_geometric as pyg
from pairing.data import PairData, Dataset, loader
from main import MixturePredictor, GCN
import tqdm

best_trial = "28b69c1c"

ms = torch.load(f"runs/{best_trial}/model.pt",map_location=torch.device('cpu'))
# SOOOOO HACKY but we saved models directly. When loading from file the model params are not saved
# So we build a new model with the same params and then build.
model = MixturePredictor(embedding_size=843,num_linear=2,num_convs=1,aggr_steps=7,architecture="GIN")
model.load_state_dict(ms.state_dict())
test_loader = loader(Dataset(is_train=False), batch_size=128)
device = "cpu"

def collate_test():
    model.eval()
    preds = []
    ys = []
    for batch in tqdm.tqdm(test_loader):
        batch.to(device)
        with torch.no_grad():
            pred = model(**batch.to_dict())

        preds.append(pred)
        ys.append(batch.y)

    return torch.cat(preds, dim=0), torch.cat(ys, dim=0)
