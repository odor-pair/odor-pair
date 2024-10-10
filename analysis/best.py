import torch
import torch_geometric as pyg
from pairing.data import PairData, Dataset, loader
from main import MixturePredictor, GCN
import tqdm
import graph.folds

def get_model():
    best_trial = "9a72dea6"
    model = MixturePredictor(embedding_size=100,num_linear=2,num_convs=6,aggr_steps=2,architecture="GIN",num_classes=Dataset.num_classes())
    model.load_state_dict(torch.load(f"runs/{best_trial}/model_state_dict.pt",map_location=torch.device('cpu')))
    model.eval()
    return model

def collate_test(fold_dataset):
    model = get_model()
    device = "cpu"
    test_loader = loader(fold_dataset["test"], batch_size=128)
    preds = []
    ys = []
    for batch in tqdm.tqdm(test_loader):
        batch.to(device)
        with torch.no_grad():
            pred = model(**batch.to_dict())

        preds.append(pred)
        ys.append(batch.y)

    return torch.cat(preds, dim=0), torch.cat(ys, dim=0)