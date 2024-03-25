import torch
from src.utils.losses import *
x = torch.rand((5,5))
idx = torch.tensor([0,1,2])

print(x,idx)

print(x[:idx.shape[0],:idx.shape[0]])