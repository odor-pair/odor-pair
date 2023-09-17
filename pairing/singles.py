from pairing.data import PairData, Dataset, loader
import pairing.data
import torch_geometric as pyg
import tqdm


all_notes = pairing.data.get_all_notes()
singles = pairing.data.get_singles()

# With very clever dictionary comprehension this could be easier.
def get_data_s(data):
    notes_s = singles[data.smiles_s]
    y_s = pairing.data.multi_hot(singles[data.smiles_s],all_notes)
    data_s = pyg.data.Data(x=data.x_s,edge_index=data.edge_index_s,edge_attr=data.edge_attr_s,y=y_s,smiles=data.smiles_s)
    return data_s

def get_data_t(data):
    notes_t = singles[data.smiles_t]
    y_t = pairing.data.multi_hot(singles[data.smiles_t],all_notes)
    data_t = pyg.data.Data(x=data.x_t,edge_index=data.edge_index_t,edge_attr=data.edge_attr_t,y=y_t,smiles=data.smiles_t)
    return data_t

def build_dataset(is_train):
    pair_data = Dataset(is_train=is_train)

    all_data = []
    seen = set()

    for d in tqdm.tqdm(pair_data):
        if not d.smiles_s in seen:
            try:
                data_s = get_data_s(d)
                all_data.append(data_s)
                seen.add(data_s.smiles)
            except AttributeError:
                pass

        if not d.smiles_t in seen:
            try:
                data_t = get_data_t(d)
                all_data.append(data_t)
                seen.add(data_t.smiles)
            except AttributeError:
                pass

    return all_data

train_data = build_dataset(True)
test_data = build_dataset(False)
print(f"Built train dataset of len = {len(train_data)}")
print(f"Built test dataset of len = {len(test_data)}")


