import torch
import numpy as np
from src.utils.data_utils import *
import datetime as dt

x = torch.rand((5,3))
q = torch.tensor([[0,1,0,1],[1,0,0,0],[0,0,0,1],[1,0,1,0]])
z = np.transpose(np.array([[0,0,2],[1,3,3]]))

edges = torch.tensor([[0,0,1],[1,2,3]])
preds = torch.argmax(x, dim=1)

y = torch.tensor([[0,1,2,2,2]])


date = dt.datetime.date(dt.datetime.now())
t = 0.8435
b = 0.6455
print('t: {:.2f}, b: {:.2f}'.format(t,b))