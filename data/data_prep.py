from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import GNNBenchmarkDataset
import pickle as pkl
# not full data set !!!
path="./"
dataset_name = 'cora'
dataset = Planetoid(root=path, name=dataset_name)

data = dataset[0]

names = ['x', 'y', 'edge_index', 'train_mask', 'val_mask', 'test_mask']

file_path = '{}/{}/dataset.pkl'.format(path,dataset_name)
try:
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)
except Exception as e:
    print("Error occurred while saving pickle file:", e)