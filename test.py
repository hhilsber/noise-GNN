import torch
from src.utils.data_utils import *
x = torch.rand((5,5))
idx = torch.tensor([[0,1,1,0],[1,2,1,0],[0,1,0,0],[0,0,0,0]])

print(idx)
adjn = normalize_adj_matrix(idx,4)