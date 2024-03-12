import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

is_cuda = torch.cuda.is_available()
print("Cuda?  is_available: {} --- version: {}".format(is_cuda,torch.version.cuda))

# GPU or CPU
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("device: {}".format(device))