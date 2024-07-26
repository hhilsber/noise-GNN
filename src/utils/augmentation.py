import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sp
import torch.nn.functional as F
from .utils import *


def topk_rewire(h, edge_index, device='cpu', k_percent=0.1, directed=False):
    if directed:
        nbr_nodes,_ = h.shape
        k = int(nbr_nodes * k_percent)
        normalized_h = F.normalize(h, dim=1)
        sim_mat = torch.mm(normalized_h, torch.transpose(normalized_h,dim0=0,dim1=1))
        
        adj = torch.zeros((nbr_nodes,nbr_nodes))
        adj[edge_index[0],edge_index[1]] = 1
        adj_remove = adj - torch.eye(nbr_nodes)
        adj_remove[(adj - torch.eye(nbr_nodes)) <= 0] = 1000
        adj_remove = adj_remove.to(device)
        
        # Remove
        values, indices = torch.topk((torch.mul(sim_mat,adj_remove)).view(-1), k=k, largest=False)
        delete_mask = torch.ones((nbr_nodes,nbr_nodes))
        delete_mask.view(-1)[indices] = 0
        adj_removed = (adj * delete_mask).to(device)

        # Add
        values, indices = torch.topk((sim_mat - adj_removed * 100 - torch.eye(nbr_nodes).to(device) * 100).view(-1), k=k)
        adj_add = torch.zeros((nbr_nodes,nbr_nodes))
        adj_add.view(-1)[indices] = 1
        adj_new = adj_removed + adj_add.to(device)
        
        indexes = torch.nonzero(adj_new, as_tuple=True)
        edge_new = torch.stack(indexes)
    else:
        nbr_nodes,_ = h.shape
        k = int(nbr_nodes * k_percent)
        normalized_h = F.normalize(h, dim=1)
        sim_mat = torch.mm(normalized_h, torch.transpose(normalized_h,dim0=0,dim1=1))
        
        adj = torch.zeros((nbr_nodes,nbr_nodes))
        adj[edge_index[0],edge_index[1]] = 1
        adj_pot = adj - torch.eye(nbr_nodes)
        adj_pot[adj_pot <= 0] = 1000
        adj_pot = adj_pot.to(device)

        # Pos edge matrix, remove worst edge
        values, indices = torch.topk((torch.mul(sim_mat,adj_pot)).view(-1), k=2*k, largest=False)
        delete_mask = torch.ones((nbr_nodes,nbr_nodes))
        delete_mask.view(-1)[indices] = 0
        adj_removed = (adj * delete_mask).to(device)
        
        # Pos edge matrix, add best edge
        values, indices = torch.topk((sim_mat - adj_removed * 100 - torch.eye(nbr_nodes).to(device) * 100).view(-1), k=2*k)
        adj_add = torch.zeros((nbr_nodes,nbr_nodes))
        adj_add.view(-1)[indices] = 1
        pos_adj = adj_removed + adj_add.to(device)

        ####################################################################################
        adj_pot = torch.zeros((nbr_nodes,nbr_nodes))
        adj_pot[edge_index[0],edge_index[1]] = 1
        adj_pot = adj_pot - torch.eye(nbr_nodes)*1000
        adj_pot = adj_pot.to(device)
        # Neg edge matrix, remove best edge
        values, indices = torch.topk((torch.mul(sim_mat,adj_pot)).view(-1), k=2*k)
        delete_mask = torch.ones((nbr_nodes,nbr_nodes))
        delete_mask.view(-1)[indices] = 0
        adj_removed = (adj * delete_mask).to(device)
        
        adj_pot = torch.ones((nbr_nodes,nbr_nodes))
        adj_pot[edge_index[0],edge_index[1]] = 1000
        adj_pot = adj_pot + torch.eye(nbr_nodes)*1000
        adj_pot = adj_pot.to(device)
        # Neg edge matrix, add worst edge
        values, indices = torch.topk((torch.mul(sim_mat,adj_pot)).view(-1), k=2*k, largest=False)
        adj_add = torch.zeros((nbr_nodes,nbr_nodes))
        adj_add.view(-1)[indices] = 1
        neg_adj = adj_removed + adj_add.to(device)

        
        indexes = torch.nonzero(pos_adj, as_tuple=True)
        pos_edge = torch.stack(indexes)
        indexes = torch.nonzero(neg_adj, as_tuple=True)
        neg_edge = torch.stack(indexes)
    return pos_edge, neg_edge

def shuffle_pos(features, device='cpu', prob=0.1):
    """
    https://github.com/TaiHasegawa/DEGNN/blob/main/models/DEGNN.py
    Augment node features for the positive graph by shuffling elements in the node feature matrix in each row.
    """
    features_shuffled = features.clone().detach()
    num_elements_to_shuffle = int(features_shuffled.shape[1] * prob)
    for i in range(features_shuffled.shape[0]):
        # get indices to be shuffled
        indices = torch.randperm(features_shuffled.shape[1])[:num_elements_to_shuffle]
        # shuffle selected element
        selected_elements = features_shuffled[i, indices]
        features_shuffled[i, indices] = selected_elements[torch.randperm(selected_elements.shape[0])]
    #print('pos feature shuffled: {}'.format(num_elements_to_shuffle))
    return features_shuffled.to(device)
    
def shuffle_neg(features, device='cpu'):
    """
    https://github.com/TaiHasegawa/DEGNN/blob/main/models/DEGNN.py
    Augment node features for the negative graph by shuffling the rows of the node feature matrix.
    """
    features_shuffled = features.clone().detach()
    idx = np.random.permutation(features.shape[0])
    features_shuffled = features_shuffled[idx, :]
    print('neg feature shuffled')
    return features_shuffled.to(device)



def augment_edges_pos(edge_index, nbr_nodes, prob=0.1):
    # Remove diag edges (a,a)
    not_equal_columns = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, not_equal_columns]

    nbr_edges_init, nbr_edges = int(edge_index.shape[1] * 0.5), int(edge_index.shape[1] * 0.5)
    nbr_del_add_init, nbr_delete, nbr_add = int(prob * nbr_edges), int(prob * nbr_edges), int(prob * nbr_edges)
    true_nbr_delete, true_nbr_add = 0, 0
    
    small_edge = torch.clone(edge_index)
    while (true_nbr_delete < nbr_del_add_init):
        #print('nbr_edges_init {}, true_nbr_delete {}, nbr_del_add_init {}'.format(nbr_edges_init, true_nbr_delete, nbr_del_add_init))
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
    print('pos augmentation: removed {} edges'.format(nbr_del_add_init))

    edge_sp = to_scipy_sparse_matrix(edge_index, nbr_nodes)
    small_sp = to_scipy_sparse_matrix(small_edge, nbr_nodes)
    big_edge = torch.tensor([[],[]])
    while (true_nbr_add != nbr_del_add_init):
        #print('nbr_edges_init {}, true_nbr_add {}, nbr_del_add_init {}'.format(nbr_edges_init, true_nbr_add, nbr_del_add_init))
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
    print('pos augmentation: added {} edges'.format(nbr_del_add_init))

    final_coo = (small_sp + big_sp).tocoo()
    row, col, data = final_coo.row, final_coo.col, final_coo.data
    final_edge = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    return final_edge.long()

def augment_edges_neg(edge_index, nbr_nodes):
    # Remove diag edges (a,a)
    not_equal_columns = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, not_equal_columns]

    nbr_edges_init, nbr_add = int(edge_index.shape[1] * 0.5), int(edge_index.shape[1] * 0.5)
    true_nbr_add = 0
    
    edge_sp = to_scipy_sparse_matrix(edge_index, nbr_nodes)
    neg_edge = torch.tensor([[0],[0]])
    while (true_nbr_add != nbr_edges_init):
        #print('nbr_edges_init {}, true_nbr_add {}'.format(nbr_edges_init, true_nbr_add))
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
    #print('neg augmentation {} edges, original graph had {}'.format(true_nbr_add, nbr_edges_init))
    return neg_edge.long()