import torch
import numpy as np
from src.utils.data_utils import *
import datetime as dt
import matplotlib.pyplot as plt


x = torch.rand((5,3))
q = torch.tensor([[0,1,0,1],[1,0,0,0],[0,0,0,1],[1,0,1,0]])
z = np.transpose(np.array([[0,0,2],[1,3,3]]))

edges = torch.tensor([[0,0,1],[1,2,3]])
preds = torch.argmax(x, dim=1)

y = torch.tensor([[0,1,2,2,2]])

train_acc_1_hist = [1,2,3]
train_acc_2_hist = [1,2,3]
val_acc_1_hist = [1,2,3]
val_acc_2_hist = [1,2,3]

pure_ratio_1_list = [4,5,6]
pure_ratio_2_list = [1,2,3]

plt.figure()
plt.subplot(211)
plt.plot(train_acc_1_hist, 'blue', label="train_acc_1_hist")
plt.plot(train_acc_2_hist, 'red', label="train_acc_2_hist")
plt.plot(val_acc_1_hist, 'green', label="val_acc_1_hist")
plt.plot(val_acc_2_hist, 'darkorange', label="val_acc_2_hist")
plt.legend()
plt.subplot(212)
plt.plot(pure_ratio_1_list, 'blue', label="pure_ratio_1_list")
plt.plot(pure_ratio_2_list, 'red', label="pure_ratio_2_list")
plt.legend()
#plt.show()
date = dt.datetime.date(dt.datetime.now())
name = '../plots/coteaching/tttt'
plt.savefig(name)