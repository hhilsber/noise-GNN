import torch
 
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data


edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

loader = DataLoader(data, batch_size=1, shuffle=True)
for batch in loader:
    batch
    
"""


 
# make some toy data
x1 = torch.Tensor([[1], [2], [3],[1], [2], [3]])
edge_index1 = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1,0,3]])
 
x2 = torch.Tensor([[1], [2]])
edge_index2 = torch.tensor([[0, 1, 1], [1, 0, 1]])
 
# make Data object
data1 = Data(x=x1, edge_index=edge_index1)
data2 = Data(x=x2, edge_index=edge_index2)
 
# dataloader
my_loader = DataLoader(data1, batch_size=2, shuffle=False)
for batch in my_loader:
    print (batch)"""