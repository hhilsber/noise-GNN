import torch
from src.utils.data_utils import *
x = torch.rand((5,5))
idx = torch.tensor([[0,1,1,0],[1,2,1,0],[0,1,0,0],[0,0,0,0]])
similarity = torch.tensor([[0,1,1,5],[1,2,1,0],[0,1,4,0],[3,0,0,0]])

#adjn = normalize_adj_matrix(idx,4)
new_adj = torch.where(similarity > 2.5, similarity, 0)
print(new_adj)