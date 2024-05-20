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

def augment_edges(edge_index, nbr_nodes, p=0.2):
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

        # Make sure we dont add new edges, and also no deleted edges
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
    

def augment_adj_old(edge_index, nbr_nodes, noise_ratio=0.2, device='cpu'):
    """
    Augment edges
    """
    nbr_edges =  edge_index.shape[1]
    nbr_edge_add_del = int(noise_ratio * (nbr_edges // 2))
    ind_delete = torch.randperm(nbr_edges // 2)[:nbr_edge_add_del] * 2
    
    mask = torch.ones(nbr_edges, dtype=torch.bool)
    mask[ind_delete] = False
    mask[ind_delete + 1] = False
    delete_edges = edge_index[:, mask]
    
    # add
    add_edges = set()
    existing_edges = set(map(tuple, edge_index.t().tolist()))
    
    while len(add_edges) < nbr_edge_add_del:
        a, b = torch.randint(0, nbr_nodes, (2,)).tolist()
        new_edge = (a, b)
        reverse_edge = (b, a)
        if new_edge not in existing_edges and new_edge not in add_edges:
            add_edges.add(new_edge)
            
    add_edges_tensor = torch.tensor(list(add_edges)).t()
    reverse_edges_tensor = add_edges_tensor.flip(0)
    
    interleaved_new_edges = torch.empty((2, 2 * nbr_edge_add_del), dtype=add_edges_tensor.dtype)
    interleaved_new_edges[:, 0::2] = add_edges_tensor
    interleaved_new_edges[:, 1::2] = reverse_edges_tensor

    perturbed_edge = torch.cat((delete_edges, interleaved_new_edges), dim=1)
    return perturbed_edge