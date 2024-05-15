import numpy as np
import torch_geometric.utils as tg
import torch
import torch.nn as nn
import torch.nn.functional as F

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
