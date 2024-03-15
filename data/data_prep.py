from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import GNNBenchmarkDataset

# not full data set !!!
path="./"
#dataset = Planetoid(root=path, name='cora')
dataset = GNNBenchmarkDataset(root=path, name='MNIST')