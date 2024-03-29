import torch
import numpy as np
from src.utils.noise import *

x = torch.rand((3,3))
y = torch.tensor([[0,1,0,1],[1,0,0,0],[0,0,0,1],[1,0,1,0]])
z = np.transpose(np.array([[0,0,2],[1,3,3]]))
#print(z)
#y[tuple(z.T)] = 0
print(x)
newx = add_feature_noise(x,0.5)
print(newx)
