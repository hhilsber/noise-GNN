from torch_geometric.datasets import Planetoid

# not full data set !!!
path="./"
dataset = Planetoid(root=path, name='cora')