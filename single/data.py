from pairing.data import PairData
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
    y = pairing.data.multi_hot(singles[data.smiles_s],all_notes)
    data_s = pyg.data.Data(x=data.x_s,edge_index=data.edge_index_s,edge_attr=data.edge_attr_s,y=y.float(),smiles=data.smiles_s)
    return data_s

def get_data_t(singles,all_notes,data):
    notes_t = singles[data.smiles_t]
    y = pairing.data.multi_hot(singles[data.smiles_t],all_notes)
    data_t = pyg.data.Data(x=data.x_t,edge_index=data.edge_index_t,edge_attr=data.edge_attr_t,y=y.float(),smiles=data.smiles_t)
    return data_t

def build_dataset(is_train):
    pair_data = pairing.data.Dataset(is_train=is_train)
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

def save(data_list, fname):
    data, slices = pyg.data.InMemoryDataset.collate(data_list)
    torch.save((data, slices), os.path.join(out_dir, fname))

def build():
    all_notes = pairing.data.get_all_notes()
    singles = pairing.data.get_singles()

    train_data_list = build_dataset(True)
    test_data_list = build_dataset(False)

    print(f"Built train dataset of len = {len(train_data_list)}")
    save(train_data_list,train_fname)

    print(f"Built test dataset of len = {len(test_data_list)}")
    save(test_data_list,test_fname)

# TODO: Refactor this into the pairing.data.Dataset
class Dataset(pyg.data.InMemoryDataset):
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

if __name__ == "__main__":
    build()

