import numpy as np
import torch
import torch.nn as nn

def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p, q):
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

class BCEExeprtLoss(nn.Module):
    """
    https://github.com/TaiHasegawa/DEGNN/blob/main/models/degnn_layers.py
    Binary cross-entropy loss for the expert.
    """
    def __init__(self, nbr_nodes):
        super(BCEExeprtLoss, self).__init__()
        #self.lbl_pos = torch.ones(nbr_nodes*1)
        #self.lbl_neg = torch.ones(nbr_nodes*1)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, logits_p, logits_n):
        #logits_pos = torch.squeeze(torch.cat((logits_p), dim=0))
        logits_pos = torch.squeeze(logits_p)
        logits_neg = torch.squeeze(logits_n)
        loss = self.criterion(logits_pos, torch.ones_like(logits_pos)) + self.criterion(logits_neg, torch.ones_like(logits_neg))
        return loss


class Discriminator_innerprod(nn.Module):
    """
    https://github.com/TaiHasegawa/DEGNN/blob/main/models/degnn_layers.py
    Discriminator defined by inner product function.
    """
    def __init__(self):
        super(Discriminator_innerprod, self).__init__()

    def forward(self, H, Hp, Hn):
        logits_p = torch.sum(torch.mul(H, Hp), dim=1, keepdim=True)
        logits_n = torch.sum(torch.mul(H, Hn), dim=1, keepdim=True)
        return logits_p, logits_n