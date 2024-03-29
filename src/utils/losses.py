import torch
import torch.nn as nn

from .data_utils import compute_degree_matrix, compute_laplacian_matrix

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

class GRTLoss(nn.Module):
    """
    Graph Regularization Technique Loss
    """
    def __init__(self, nbr_nodes, alpha, beta, gamma, device):
        super(GRTLoss, self).__init__()
        self.n = nbr_nodes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device
    
    def forward(self, A, X):
        D = compute_degree_matrix(A)
        L = compute_laplacian_matrix(D, A)

        smoothness = compute_smoothness(X, L, self.n)
        connectivity = compute_connectivity(A, self.n, self.device)
        sparsity = compute_sparsity(A, self.n)
        #print("smoothness {} connectivity {}  sparsity {}".format(smoothness,connectivity,sparsity))
        #loss = self.alpha * smoothness + self.beta * connectivity + self.gamma * sparsity
        #return loss
        return smoothness, connectivity, sparsity

def compute_smoothness(X, L, nbr_nodes):
    XtL = torch.matmul(torch.transpose(X, 0, 1), L)
    smoothness = (1/(nbr_nodes*nbr_nodes)) * torch.trace(torch.matmul(XtL, X))
    return smoothness
def compute_connectivity(A, nbr_nodes, device):
    #log_A1 = torch.log(torch.matmul(A, torch.ones(nbr_nodes).long()))
    #print(A)
    log_A1 = torch.log(torch.matmul(A, torch.ones(nbr_nodes).to(device)))
    #print(log_A1)
    connectivity = (-1/nbr_nodes) * torch.matmul(torch.ones((1,nbr_nodes)).to(device), log_A1)
    return connectivity
def compute_sparsity(A, nbr_nodes):
    sparsity = (1/(nbr_nodes*nbr_nodes)) * torch.pow(torch.linalg.norm(A.float(), ord='fro'), exponent=2)
    return sparsity