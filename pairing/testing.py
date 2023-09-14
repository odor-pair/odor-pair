import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# The ordering of the graphs in this is arbitrary (alphabetically based on SMILES)
# but the logistic predictor does rely on this ordering.
class PairData(Data):

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


x_1 = torch.randn(5, 16)  # 5 nodes.
edge_index_1 = torch.tensor([
    [0, 0, 0, 0],
    [1, 2, 3, 4],
])

x_2 = torch.randn(4, 16)  # 4 nodes.
edge_index_2 = torch.tensor([
    [0, 0, 0],
    [1, 2, 3],
])

x_3 = torch.randn(3, 16)  # 5 nodes.
edge_index_3 = torch.tensor([
    [0, 0, 1, 1],
    [1, 2, 2, 0],
])

x_4 = torch.randn(2, 16)  # 5 nodes.
edge_index_4 = torch.tensor([
    [0, 1],
    [1, 0],
])

data1 = PairData(x_s=x_1,
                 edge_index_s=edge_index_1,
                 x_t=x_2,
                 edge_index_t=edge_index_2)
data2 = PairData(x_s=x_3,
                 edge_index_s=edge_index_3,
                 x_t=x_4,
                 edge_index_t=edge_index_4)

data_list = [data1, data2, data1]
loader = DataLoader(data_list, batch_size=3, follow_batch=['x_s', 'x_t'])
batch = next(iter(loader))

print(batch.x_s_batch)
