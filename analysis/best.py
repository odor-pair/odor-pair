import torch
import torch_geometric as pyg
from pairing.data import PairData, Dataset, loader
from main import MixturePredictor, GCN
import tqdm

def get_model():
    best_trial = "1efcd2d0"

    ms = torch.load(f"runs/{best_trial}/model.pt",map_location=torch.device('cpu'))
    # SOOOOO HACKY but we saved models directly. When loading from file the model params are not saved
    # So we build a new model with the same params and then build.
    model = MixturePredictor(embedding_size=832,num_linear=2,num_convs=3,aggr_steps=0,architecture="GIN")
    model.load_state_dict(ms.state_dict())
    return model

def collate_test():
    model = get_model()

    device = "cpu"
    test_loader = loader(Dataset(is_train=False), batch_size=128)
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
