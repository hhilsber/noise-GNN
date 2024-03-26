import numpy as np
import torch_geometric.utils as tg
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric.loader import DataLoader
#from torch_geometric.data import Batch
#from torch_geometric.data import Data
from torch.utils.data import Dataset

def eval_classification(prediction, labels):
    """
    evaluate results
    """
    if len(labels.size()) == 2:
        preds = torch.round(torch.sigmoid(prediction))
        tp = len(torch.nonzero(preds * labels))
        tn = len(torch.nonzero((1-preds) * (1-labels)))
        fp = len(torch.nonzero(preds * (1-labels)))
        fn = len(torch.nonzero((1-preds) * labels))
        pre, rec, f1 = 0., 0., 0.
        if tp+fp > 0:
            pre = tp / (tp + fp)
        if tp+fn > 0:
            rec = tp / (tp + fn)
        if pre+rec > 0:
            fmeasure = (2 * pre * rec) / (pre + rec)
    else:
        preds = torch.argmax(prediction, dim=1)
        correct = torch.sum(preds == labels)
        fmeasure = correct.item() / len(labels)
    return fmeasure

def compute_degree_matrix(adj):
    return torch.diag(torch.sum(adj, dim=1).float())
def compute_laplacian_matrix(deg, adj):
    return deg-adj

def normalize_graph_laplacian(adj):
    deg = compute_degree_matrix(adj)
    D_plus = torch.linalg.pinv(deg)
    lap = compute_laplacian_matrix(deg, adj)

    norm_GL = torch.matmul(torch.pow(D_plus, 0.5), lap)
    norm_GL = torch.matmul(norm_GL, torch.pow(D_plus, 0.5))
    return norm_GL


def normalize_adj_matrix(adj_matrix, nbr_nodes, device):
    """
    normalize adj GCN
    """
    n_nodes = nbr_nodes
    adj_norm = adj_matrix
    adj_norm = adj_norm * (torch.ones(n_nodes).to(device) - torch.eye(n_nodes).to(device)) + torch.eye(n_nodes).to(device)
    D_norm = torch.diag(torch.pow(adj_norm.sum(1), -0.5)).to(device)
    adj_norm = D_norm @ adj_norm @ D_norm
    return adj_norm
    
def normalize_adj_matrix_old(adj_matrix, nbr_nodes, device):
    """
    normalize adjacency matrix
    """
    matrix = adj_matrix * (torch.ones(nbr_nodes).to(device) - torch.eye(nbr_nodes).to(device)) + torch.eye(nbr_nodes).to(device)
    degree_norm = torch.diag(torch.pow(matrix.sum(1), -0.5)).to(device)
    #degree_norm[torch.isinf(degree_norm)] = 0.
    adj_norm = torch.matmul(degree_norm, matrix)
    adj_norm = torch.matmul(adj_norm, degree_norm)
    return adj_norm

class Rewire(torch.nn.Module):
    """
    Rewire nodes using similarity matrix
    """
    def __init__(self, ratio, device='cpu'):
        super(Rewire, self).__init__()
        self.device = device
        self.bot_q = ratio
        self.top_q = 1. - ratio

    
    def forward(self, similarity, adjacency):
        #normalize similarity?
        #normalized_z = F.normalize(H, dim=1)
        sim_norm = F.normalize(similarity, dim=1)
        sim_mat = torch.mm(sim_norm, sim_norm.t()) * (torch.ones_like(sim_norm) - torch.eye(sim_norm.shape[0]))

        quant_bot = torch.quantile(sim_mat, self.bot_q)
        print('edges removed: {}'.format(quant_bot))
        quant_top = torch.quantile(sim_mat, self.top_q)
        print('edges added: {}'.format(quant_top))
        
        new_adj = torch.where(similarity <= quant_bot, 0, adjacency)
        new_adj = torch.where(similarity > quant_top, sim_mat, new_adj)
        return new_adj.to(self.device)

def create_lbl_mat(labels):
    # Create (nnode x nnode) label matrix, where (i,j) is one if label_i = label_j
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