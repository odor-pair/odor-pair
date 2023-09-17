from pairing.data import PairData, Dataset, loader
import pairing.data
import torch_geometric as pyg
import tqdm
import torch
import os


out_dir = "single"
train_fname = "trainsingles.pt"
test_fname = "testsingles.pt"

# With very clever dictionary comprehension this could be easier.
def get_data_s(singles,all_notes,data):
    notes_s = singles[data.smiles_s]
    y_s = pairing.data.multi_hot(singles[data.smiles_s],all_notes)
    data_s = pyg.data.Data(x=data.x_s,edge_index=data.edge_index_s,edge_attr=data.edge_attr_s,y=y_s,smiles=data.smiles_s)
    return data_s

def get_data_t(singles,all_notes,data):
    notes_t = singles[data.smiles_t]
    y_t = pairing.data.multi_hot(singles[data.smiles_t],all_notes)
    data_t = pyg.data.Data(x=data.x_t,edge_index=data.edge_index_t,edge_attr=data.edge_attr_t,y=y_t,smiles=data.smiles_t)
    return data_t

def build_dataset(is_train):
    pair_data = Dataset(is_train=is_train)
    singles = pairing.data.get_singles()
    all_notes = pairing.data.get_all_notes()

    all_data = []
    seen = set()

    for d in tqdm.tqdm(pair_data):
        if not d.smiles_s in seen:
            try:
                data_s = get_data_s(singles,all_notes,d)
                all_data.append(data_s)
                seen.add(data_s.smiles)
            except AttributeError:
                pass

        if not d.smiles_t in seen:
            try:
                data_t = get_data_t(singles,all_notes,d)
                all_data.append(data_t)
                seen.add(data_t.smiles)
            except AttributeError:
                pass

    return all_data

def build():
    all_notes = pairing.data.get_all_notes()
    singles = pairing.data.get_singles()

    train_data_list = build_dataset(True)
    test_data_list = build_dataset(False)

    print(f"Built train dataset of len = {len(train_data_list)}")
    pairing.data.save(train_data_list,train_fname)

    print(f"Built test dataset of len = {len(test_data_list)}")
    pairing.data.save(train_data_list,test_fname)

# TODO: Refactor this into the pairing.data.Dataset
class SinglesDataset(pyg.data.InMemoryDataset):
    def __init__(self, is_train):
        super().__init__(out_dir)
        if is_train:
            self.data, self.slices = torch.load(
                os.path.join(out_dir, train_fname))
        else:
            self.data, self.slices = torch.load(
                os.path.join(out_dir, test_fname))

    @classmethod
    def num_classes(cls):
        return pairing.data.Dataset.num_classes()

    @classmethod
    def num_node_features(cls):
        return pairing.data.Dataset.num_node_features()

    @classmethod
    def num_edge_features(cls):
        return pairing.data.Dataset.num_edge_features()

build()

