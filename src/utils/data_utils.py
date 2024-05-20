import numpy as np
import torch_geometric.utils as tg
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sp

def to_scipy_sparse_matrix(edge_index, num_nodes):
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.ones(len(row))
    shape = (num_nodes, num_nodes)
    return sp.coo_matrix((data, (row, col)), shape=shape)

def to_scipy_sparse_matrix_rcd(row, col, data, num_nodes):
    shape = (num_nodes, num_nodes)
    return sp.coo_matrix((data, (row, col)), shape=shape)

def augment_edges(edge_index, nbr_nodes, p=0.05):
    # Remove diag edges (a,a)
    not_equal_columns = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, not_equal_columns]

    nbr_edges_init, nbr_edges = int(edge_index.shape[1] * 0.5), int(edge_index.shape[1] * 0.5)
    nbr_del_add_init, nbr_delete, nbr_add = int(p * nbr_edges), int(p * nbr_edges), int(p * nbr_edges)
    true_nbr_delete, true_nbr_add = 0, 0
    
    small_edge = torch.clone(edge_index)
    while (true_nbr_delete < nbr_del_add_init):
        print('nbr_edges_init {}, true_nbr_delete {}, nbr_del_add_init {}'.format(nbr_edges_init, true_nbr_delete, nbr_del_add_init))
        # Sample edges to delete
        sample_indices = np.random.choice(nbr_edges, nbr_delete, replace=False)
        edge_delete = small_edge[:,sample_indices]
        edge_delete = torch.cat((edge_delete,torch.flip(edge_delete, dims=[0])),1)
        
        # Convert to coo matrix and delete edges
        small_sp = to_scipy_sparse_matrix(small_edge, nbr_nodes)
        del_sp = to_scipy_sparse_matrix(edge_delete, nbr_nodes)
        row, col, data = sp.find((small_sp - del_sp) > 0)
        small_edge = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
        
        # Check if we deleted enough edges, not only same edges (a,b) and (b,a)
        nbr_edges = int(small_edge.shape[1] * 0.5)
        true_nbr_delete = nbr_edges_init - nbr_edges
        nbr_delete = nbr_del_add_init - true_nbr_delete
        
    edge_sp = to_scipy_sparse_matrix(edge_index, nbr_nodes)
    small_sp = to_scipy_sparse_matrix(small_edge, nbr_nodes)
    big_edge = torch.tensor([[0],[0]])
    while (true_nbr_add != nbr_del_add_init):
        print('nbr_edges_init {}, true_nbr_add {}, nbr_del_add_init {}'.format(nbr_edges_init, true_nbr_add, nbr_del_add_init))
        # Generate edges to add
        rand_edges = torch.randint(0, nbr_nodes, (2, nbr_add))
        edge_add = torch.cat((rand_edges,torch.flip(rand_edges, dims=[0])),1)
        
        # Convert to coo matrix and add edges
        if big_edge.shape[1] >= 2:
            big_sp = to_scipy_sparse_matrix(big_edge, nbr_nodes)
            add_sp = to_scipy_sparse_matrix(edge_add, nbr_nodes)
            row, col, data = sp.find((big_sp + add_sp) == 1)
            big_sp = to_scipy_sparse_matrix_rcd(row, col, data, nbr_nodes)
        else:
            big_sp = to_scipy_sparse_matrix(edge_add, nbr_nodes)

        # Make sure we add new edges, and also no deleted edges
        row, col, data = sp.find((edge_sp + small_sp - big_sp) < 0)
        big_edge = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
        # Remove diag edges (a,a)
        not_equal_columns = big_edge[0] != big_edge[1]
        big_edge = big_edge[:, not_equal_columns]

        # Check if we added enough edges
        true_nbr_add = int(big_edge.shape[1] * 0.5)
        nbr_add = nbr_del_add_init - true_nbr_add
    final_coo = (small_sp + big_sp).tocoo()
    row, col, data = final_coo.row, final_coo.col, final_coo.data
    final_edge = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    return final_edge

def augment_edges_neg(edge_index, nbr_nodes):
    # Remove diag edges (a,a)
    not_equal_columns = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, not_equal_columns]

    nbr_edges_init, nbr_add = int(edge_index.shape[1] * 0.5), int(edge_index.shape[1] * 0.5)
    true_nbr_add = 0
    
    edge_sp = to_scipy_sparse_matrix(edge_index, nbr_nodes)
    neg_edge = torch.tensor([[0],[0]])
    while (true_nbr_add != nbr_edges_init):
        print('nbr_edges_init {}, true_nbr_add {}'.format(nbr_edges_init, true_nbr_add))
        # Generate edges to add
        rand_edges = torch.randint(0, nbr_nodes, (2, nbr_add))
        edge_add = torch.cat((rand_edges,torch.flip(rand_edges, dims=[0])),1)
        
        # Convert to coo matrix and add edges
        if neg_edge.shape[1] >= 2:
            neg_sp = to_scipy_sparse_matrix(neg_edge, nbr_nodes)
            add_sp = to_scipy_sparse_matrix(edge_add, nbr_nodes)
            row, col, data = sp.find((neg_sp + add_sp) == 1)
            neg_sp = to_scipy_sparse_matrix_rcd(row, col, data, nbr_nodes)
        else:
            neg_sp = to_scipy_sparse_matrix(edge_add, nbr_nodes)

        # Make sure we add edges different from original graph
        row, col, data = sp.find((edge_sp - neg_sp) < 0)
        neg_edge = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
        # Remove diag edges (a,a)
        not_equal_columns = neg_edge[0] != neg_edge[1]
        neg_edge = neg_edge[:, not_equal_columns]

        # Check if we added enough edges
        true_nbr_add = int(neg_edge.shape[1] * 0.5)
        nbr_add = nbr_edges_init - true_nbr_add
    return neg_edge

def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

def topk_accuracy(output, target, batch_size, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #output = F.softmax(logit, dim=1)
    maxk = max(topk)

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    #pred = pred.t()
    #correct = pred.eq(target.view(1, -1).expand_as(pred))
    print(output.shape, pred.shape, target.shape)
    correct = pred.eq(target)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def augment_features(features, noise_ratio, device='cpu'):
    """
    https://github.com/TaiHasegawa/DEGNN/blob/main/models/DEGNN.py
    Augment node features for the positive graph by shuffling elements in the node feature matrix in each row.
    """
    features_shuffled = features.clone().detach()
    num_elements_to_shuffle = int(features_shuffled.shape[1] * noise_ratio)
    for i in range(features_shuffled.shape[0]):
        # get indices to be shuffled
        indices = torch.randperm(features_shuffled.shape[1])[:num_elements_to_shuffle]
        # shuffle selected element
        selected_elements = features_shuffled[i, indices]
        features_shuffled[i, indices] = selected_elements[torch.randperm(selected_elements.shape[0])]
    return features_shuffled.to(device)
    
