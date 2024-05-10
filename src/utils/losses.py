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

        return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2

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
        ind_1_sorted = np.argsort(loss_1.data)
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, y_noise, reduction='none') - self.co_lambda * js_loss_2
        #loss_2 = F.cross_entropy(y_2, y_noise, reduction='none') - self.co_lambda * self.js_loss_compute(y_1, y_2, reduce=False)
        ind_2_sorted = np.argsort(loss_2.data)
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        if len(ind_1_update) == 0:
            ind_1_update = ind_1_sorted.numpy()
            ind_2_update = ind_2_sorted.numpy()
            num_remember = ind_1_update.shape[0]

        pure_ratio_1 = torch.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
        pure_ratio_2 = torch.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

        loss_1_update = torch.sum(loss_1[ind_2_update])/num_remember
        loss_2_update = torch.sum(loss_2[ind_1_update])/num_remember
        
        return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2

        #loss_1_update = torch.sum(loss_1[ind_2_update])/num_remember
        #loss_2_update = torch.sum(loss_2[ind_1_update])/num_remember
        #loss_1_update = torch.sum(torch.index_select(loss_1, 0, ind_2_update)) / num_remember
        #loss_2_update = torch.sum(torch.index_select(loss_2, 0, ind_1_update)) / num_remember
        #loss_1_update = torch.sum(torch.masked_select(loss_1, ind_2_update.bool())) / num_remember
        #loss_2_update = torch.sum(torch.masked_select(loss_2, ind_1_update.bool())) / num_remember