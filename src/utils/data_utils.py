import numpy as np
import torch_geometric.utils as tg
import torch
import torch.nn as nn
#from torch_geometric.loader import DataLoader
#from torch_geometric.data import Batch
#from torch_geometric.data import Data
from torch.utils.data import Dataset

class BCELoss(nn.Module):
    """
    Binary cross-entropy loss
    """
    def __init__(self, label_mat, device):
        super(BCELoss, self).__init__()
        self.label_mat = label_mat.to(device)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, output):
        loss = self.criterion(output, self.label_mat)
        return loss

class NormalDataset(Dataset):
    """
    Class to save the training dataset 
    """
    def __init__(self, features, adjacency, edge_list=None):
        # Store images and groundtruths
        self.feat, self.adj = features, adjacency

    def __len__(self): 
        # Returns len (used for data loaders) 
        return len(self.feat)

    def __getitem__(self, idx=-1):
        # Return dataset
        return self.feat.float(), self.adj

def create_lbl_mat(labels):
    lbl_mat = np.zeros((labels.shape[0],labels.shape[0]))
    for lbl in range(labels.max().item() + 1):
        same_lbl = np.where(labels==lbl)[0]
        for i in same_lbl:
            for j in same_lbl:
                lbl_mat[i,j] = 1
    return torch.from_numpy(lbl_mat).float()


def sample_neighbor(dataset, k_hop=2):
    nnbr = 0 # node number
    nlbl = dataset['labels'][nnbr] # node label
    subset, edge_index, mapping, edge_mask = tg.k_hop_subgraph(node_idx=nnbr, num_hops=k_hop, edge_index=dataset['edge_list'], relabel_nodes=True)
    
    same_class = np.where(dataset['labels']==nlbl)[0]
    print("  subset: {}".format(subset))
    print("  same_class: {}".format(same_class))
    inter = np.intersect1d(subset.tolist(), same_class)
    print("  intersect: {}".format(inter))
    res = [item for item in subset.tolist() if item not in inter]
    print("  diff inter: {}".format(res))

"""
class OldDataset(Dataset):
    def __init__(self, features, adjacency):
        # Store images and groundtruths
        self.feat, self.adj = features, adjacency

    def __len__(self): 
        # Returns len (used for data loaders) 
        return len(self.feat)

    def __getitem__(self, idx=-1):
        # Return dataset
        return self.feat[idx].float(), self.adj[idx]"""