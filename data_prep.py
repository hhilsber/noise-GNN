from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import GNNBenchmarkDataset
import pickle as pkl
from ogb.nodeproppred import PygNodePropPredDataset


path="./"
dataset_name = 'ogbn-arxiv'

if dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name = dataset_name)
        #graph = dataset[0]

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        dataset.train_idx = train_idx
        dataset.valid_idx = valid_idx
        dataset.test_idx = test_idx
        
        print(dataset.x.shape)
        print(split_idx["train"].shape)
        print(max(split_idx["train"]))
        print(len(dataset[0]))
        ttt = dataset[split_idx["train"]]
        

elif dataset_name == 'cora':
    data = Planetoid(root=path, name=dataset_name)

    dataset = data[0]

    names = ['x', 'y', 'edge_index', 'train_mask', 'val_mask', 'test_mask']

"""
file_path = '{}/{}/dataset.pkl'.format(path,dataset_name)
try:
    with open(file_path, 'wb') as f:
        pkl.dump(dataset, f)
except Exception as e:
    print("Error occurred while saving pickle file:", e)"""