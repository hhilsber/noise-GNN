import torch
import numpy as np
from src.utils.data_utils import *

x = torch.rand((3,3))
y = torch.tensor([[0,1,0,1],[1,0,0,0],[0,0,0,1],[1,0,1,0]])
z = np.transpose(np.array([[0,0,2],[1,3,3]]))

edges = torch.tensor([[0,0,1],[1,2,3]])

adj = edges_to_adjacency(edges, 4)
print(adj)