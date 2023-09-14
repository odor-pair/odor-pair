import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from ogb.utils import smiles2graph
import pandas as pd
import tqdm

fname = "aqsol/data.pt"


def to_torch(smiles, y):
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    graph = smiles2graph(smiles)
    tensor_keys = ["edge_index", 'edge_feat', 'node_feat']
    for key in tensor_keys:
        graph[key] = torch.tensor(graph[key])
    return Data(x=graph["node_feat"].float(),
                edge_index=graph["edge_index"],
                edge_attr=graph["edge_feat"],
                y=y)


# SOOOOO HACKY. Call this to rebuild
def build():
    df = pd.read_csv("aqsol/curated-solubility-dataset.csv")
    data_list = []
    for index, row in tqdm.tqdm(df.iterrows()):
        d = to_torch(row['SMILES'], row['Solubility'])
        data_list.append(d)

    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), fname)


class Dataset(InMemoryDataset):

    def __init__(self, transform=None):
        super().__init__("aqsol", transform)
        self.data, self.slices = torch.load(fname)
