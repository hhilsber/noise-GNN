import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from .utils import *

def neighbor_align_batch(edge_index, x, h,
                    batch_mask,
                    epsilon = 1e-16,
                    temp = 0.1,
                    ncr_conf = 0.0,
                    ncr_loss = 'kl',
                    on_noisy_lbl = 'yes',
                    device = 'cpu'):

    
    p = torch.exp(h)
    
    num_nodes = h.shape[0]
    coo_matrix = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes)
    indices = np.vstack((np.array(coo_matrix.row), np.array(coo_matrix.col)))
    adj_matrix = torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(np.array(coo_matrix.data)),
        torch.Size(coo_matrix.shape)
    )

    adj_matrix = adj_matrix.to(device)
    h = h.to(device)

    if ncr_loss == 'kl':
        mean = torch.sparse.mm(adj_matrix, h)
        mean = mean / (adj_matrix.sum(dim=1).to_dense().view(-1,1) + epsilon)
        sharp_mean = (torch.pow(mean, 1./temp) / torch.sum(torch.pow(mean, 1./temp) + epsilon, dim=1, keepdim=True)).detach()
        if on_noisy_lbl == "yes":
            kl_loss = F.kl_div(h, sharp_mean, reduction='none')[batch_mask].sum(1)
            filtered_kl_loss = kl_loss[mean[batch_mask].max(1)[0] > ncr_conf]
            local_ncr = torch.mean(filtered_kl_loss)
        else:
            local_ncr = torch.mean((-sharp_mean * torch.log_softmax(h, dim=1)).sum(1)[torch.softmax(mean, dim=-1).max(1)[0] > ncr_conf])
    else:
        print('wrong ncr_loss')

    return local_ncr

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
        ind_1_sorted = np.argsort(loss_1.cpu().data)
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, y_noise, reduction = 'none')
        ind_2_sorted = np.argsort(loss_2.cpu().data)
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))
        
        pure_ratio_1 = torch.sum(noise_or_not[ind.cpu()[ind_1_sorted[:num_remember]]])/float(num_remember)
        pure_ratio_2 = torch.sum(noise_or_not[ind.cpu()[ind_2_sorted[:num_remember]]])/float(num_remember)

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        """
        ind_clean_1 = ind.cpu()[ind_1_sorted[:num_remember]]
        ind_clean_2 = ind.cpu()[ind_2_sorted[:num_remember]]
        ind_noisy_1 = ind.cpu()[ind_1_sorted[num_remember:]]
        ind_noisy_2 = ind.cpu()[ind_2_sorted[num_remember:]]"""
        batch_ind_noisy_1 = ind_1_sorted[num_remember:]
        batch_ind_noisy_2 = ind_2_sorted[num_remember:]
        # exchange
        loss_1_update = F.cross_entropy(y_1[ind_2_update], y_noise[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], y_noise[ind_1_update])

        #return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
        return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2, batch_ind_noisy_1, batch_ind_noisy_2

class CoDiLoss(nn.Module):
    """
    CoDis loss
    https://github.com/tmllab/2023_ICCV_CoDis/blob/main/loss.py
    """
    def __init__(self, device, co_lambda=0.1):
        super(CoDiLoss, self).__init__()
        self.device = device
        self.co_lambda = co_lambda
    
    def kl_loss_compute(self, pred, soft_targets, reduce=True):

        kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1), reduction=False)

        if reduce:
            return torch.mean(torch.sum(kl, dim=1))
        else:
            return torch.sum(kl, 1)
        

    def js_loss_compute(self, pred, soft_targets, reduce=True):
        
        pred_softmax = F.softmax(pred, dim=1)
        targets_softmax = F.softmax(soft_targets, dim=1)
        mean = (pred_softmax + targets_softmax) / 2
        kl_1 = F.kl_div(F.log_softmax(pred, dim=1), mean, reduction='none')
        kl_2 = F.kl_div(F.log_softmax(soft_targets, dim=1), mean, reduction='none')
        js = (kl_1 + kl_2) / 2 
        
        if reduce:
            return torch.mean(torch.sum(js, dim=1))
        else:
            return torch.sum(js, 1)

    def forward(self, y_1, y_2, y_noise, forget_rate, ind, noise_or_not):
        js_loss = self.js_loss_compute(y_1, y_2, reduce=False)
        js_loss_1 = js_loss.detach()
        js_loss_2 = js_loss.detach()
        loss_1 = F.cross_entropy(y_1, y_noise, reduction='none') - self.co_lambda * js_loss_1
        #loss_1 = F.cross_entropy(y_1, y_noise, reduction='none') - self.co_lambda * self.js_loss_compute(y_1, y_2, reduce=False)
        ind_1_sorted = np.argsort(loss_1.cpu().data)
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, y_noise, reduction='none') - self.co_lambda * js_loss_2
        #loss_2 = F.cross_entropy(y_2, y_noise, reduction='none') - self.co_lambda * self.js_loss_compute(y_1, y_2, reduce=False)
        ind_2_sorted = np.argsort(loss_2.cpu().data)
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        if len(ind_1_update) == 0:
            ind_1_update = ind_1_sorted.numpy()
            ind_2_update = ind_2_sorted.numpy()
            num_remember = ind_1_update.shape[0]

        pure_ratio_1 = torch.sum(noise_or_not[ind.cpu()[ind_1_sorted[:num_remember]]])/float(num_remember)
        pure_ratio_2 = torch.sum(noise_or_not[ind.cpu()[ind_2_sorted[:num_remember]]])/float(num_remember)
       
        loss_1_update = F.cross_entropy(y_1[ind_2_update], y_noise[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], y_noise[ind_1_update])

        #return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
        return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2

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

