import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from .utils import *

def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p+1e-5), dim=axis)

def get_uncertainty_batch(edge_index, y_pure, nbr_classes, device = 'cpu', epsilon=1e-16):

    p = torch.exp(y_pure)
    num_nodes = y_pure.shape[0]

    coo_matrix = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes)
    indices = np.vstack((np.array(coo_matrix.row), np.array(coo_matrix.col)))

    adj_matrix = torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(np.array(coo_matrix.data)),
        torch.Size(coo_matrix.shape)
    )
    adj_matrix = adj_matrix.to(device)

    ptc = torch.sparse.mm(adj_matrix, p)
    ptc = ptc / (adj_matrix.sum(dim=1).to_dense().view(-1,1) + epsilon)
    hpt = entropy(ptc)
    w = torch.exp(-hpt/torch.log2(torch.tensor(nbr_classes)))
    return w

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    if use_hard_labels:
        return F.cross_entropy(logits, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets*log_pred, dim=1)
        return nll_loss

def fix_cr(y_pure, y_noisy, ind_noisy, batch_size=512, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True, w=None):
    assert name in ['ce', 'l2']

    num_nodes = y_pure.shape[0]
    mask_noisy = np.zeros(num_nodes).astype(bool)
    mask_noisy[ind_noisy] = True

    y_pure = y_pure[:batch_size]
    y_noisy = y_noisy[:batch_size]

    pseudo_pure = torch.exp(y_pure)
    pseudo_noisy = torch.exp(y_noisy)

    #pseudo_pure = pseudo_pure.detach()
    if name == 'l2':
        assert y_pure.size() == y_noisy.size()
        return F.mse_loss(y_noisy, y_pure, reduction='mean')
    elif name == 'ce':
        # pseudo_pure = torch.softmax(y_pure, dim=-1)
        max_probs, max_idx = torch.max(pseudo_pure, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        if use_hard_labels:
            masked_loss = ce_loss(pseudo_noisy, max_idx, use_hard_labels, reduction='none') * mask
        else:
            #pseudo_pure = torch.softmax(y_pure/T, dim=-1)
            masked_loss = ce_loss(pseudo_noisy, pseudo_pure, use_hard_labels) * mask
        if w is None:
            return masked_loss.mean() #, mask.mean()
        else:
            return (w[:batch_size] * masked_loss).mean()
    else:
        assert Exception('Not Implemented consistency_loss')

def neighbor_align_batch(edge_index, x, h,
                    ind_noisy,
                    batch_size = 512,
                    epsilon = 1e-16,
                    temp = 0.1,
                    ncr_conf = 0.0,
                    ncr_loss = 'kl',
                    on_noisy_lbl = 'yes',
                    device = 'cpu'):

    num_nodes = h.shape[0]
    mask_noisy = np.zeros(num_nodes).astype(bool)
    mask_noisy[ind_noisy] = True

    coo_matrix = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes)
    indices = np.vstack((np.array(coo_matrix.row), np.array(coo_matrix.col)))
    adj_matrix = torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(np.array(coo_matrix.data)),
        torch.Size(coo_matrix.shape)
    )
    #adj_matrix_ts = torch.tensor(coo_matrix.toarray()).float().to(device)
    adj_matrix_coo = adj_matrix.to(device)
    y = torch.softmax(h, dim=1).to(device)
    h = h.to(device)
    print(h[:1,:])
    print(y[:1,:])
    #print(torch.sum(adj_matrix_ts[:batch_size,:batch_size]))
    print(a)
    if ncr_loss == 'kl':
        mean = torch.sparse.mm(adj_matrix_coo, h) #[:batch_size,:]
        #mean = torch.matmul(adj_matrix_ts, h)[:batch_size,:]
        #mean = mean / (adj_matrix_ts[:batch_size,:batch_size].sum(dim=1).to_dense().view(-1,1) + epsilon)
        mean = mean / (adj_matrix_coo.sum(dim=1).to_dense().view(-1,1) + epsilon)
        sharp_mean = (torch.pow(mean, 1./temp) / torch.sum(torch.pow(mean, 1./temp) + epsilon, dim=1, keepdim=True)).detach()
        
        if on_noisy_lbl == "yes":
            kl_loss = F.kl_div(h, sharp_mean, reduction='none')[mask_noisy].sum(1)
            #filtered_kl_loss = kl_loss[mean[:batch_size][ind_noisy].max(1)[0] > ncr_conf]
            filtered_kl_loss = kl_loss[mean[mask_noisy].max(1)[0] > ncr_conf]
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

        ind_noisy_1 = ind_1_sorted[num_remember:]
        ind_noisy_2 = ind_2_sorted[num_remember:]
        # exchange
        loss_1_update = F.cross_entropy(y_1[ind_2_update], y_noise[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], y_noise[ind_1_update])

        #return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
        return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update, ind_noisy_1, ind_noisy_2

def backward_correction(output, labels, C, nbr_class, device):
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
    label_oh = torch.FloatTensor(len(labels), nbr_class).to(device)
    label_oh.zero_()
    label_oh.scatter_(1,labels.view(-1,1),1)
    output = softmax(output)
    #output /= torch.sum(output, dim=-1, keepdim=True)
    output = torch.clamp(output, min=1e-5, max=1.0-1e-5)
    return -torch.mean(torch.matmul(label_oh, C_inv) * torch.log(output))

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



class CTLoss2(nn.Module):
    """
    Co-teaching loss
    https://github.com/bhanML/Co-teaching/blob/master/loss.py
    """
    def __init__(self, device):
        super(CTLoss2, self).__init__()
        self.device = device
    
    def forward(self, y_1, y_2, y_noise, y_noise2, forget_rate, ind, noise_or_not):
        loss_1 = F.cross_entropy(y_1, y_noise, reduction = 'none')
        ind_1_sorted = np.argsort(loss_1.cpu().data)
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, y_noise2, reduction = 'none')
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

        ind_noisy_1 = ind_1_sorted[num_remember:]
        ind_noisy_2 = ind_2_sorted[num_remember:]
        # exchange
        loss_1_update = F.cross_entropy(y_1[ind_2_update], y_noise[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], y_noise2[ind_1_update])

        #return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
        return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2, ind_noisy_1, ind_noisy_2
