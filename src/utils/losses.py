import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def backward_correction(output, labels, C, device, nclass):
    '''
    https://github.com/gear/denoising-gnn/blob/master/models/loss.py

        Backward loss correction.

        output: raw (logits) output from model
        labels: true labels
        C: correction matrix
    '''
    softmax = nn.Softmax(dim=1)
    C_inv = np.linalg.inv(C).astype(np.float32)
    C_inv = torch.from_numpy(C_inv).to(device)
    label_oh = torch.FloatTensor(len(labels), nclass).to(device)
    label_oh.zero_()
    label_oh.scatter_(1,labels.view(-1,1),1)
    output = softmax(output)
    #output /= torch.sum(output, dim=-1, keepdim=True)
    output = torch.clamp(output, min=1e-5, max=1.0-1e-5)
    return -torch.mean(torch.matmul(label_oh, C_inv) * torch.log(output))

class CTLoss(nn.Module):
    """
    Co-teaching loss
    https://github.com/bhanML/Co-teaching/blob/master/loss.py
    """
    def __init__(self, device):
        super(CTLoss, self).__init__()
        self.device = device
    
    def forward(self, y_1, y_2, y_noise, forget_rate, ind, noise_or_not):
        loss_1 = F.cross_entropy(y_1, y_noise, reduction = 'none')
        ind_1_sorted = np.argsort(loss_1.data)
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, y_noise, reduction = 'none')
        ind_2_sorted = np.argsort(loss_2.data)
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))
        
        pure_ratio_1 = torch.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
        pure_ratio_2 = torch.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

        ind_1_update=ind_1_sorted[:num_remember]
        ind_2_update=ind_2_sorted[:num_remember]
        # exchange
        loss_1_update = F.cross_entropy(y_1[ind_2_update], y_noise[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], y_noise[ind_1_update])

        #return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
        return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2

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