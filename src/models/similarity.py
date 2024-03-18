import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GAT

class Similarity(nn.Module):
    def __init__(self, config):
        super(Similarity, self).__init__()
        self.config = config
        self.name = 'Similarity'
        self.device = config['device']
        nfeat = config['nbr_features']
        nclass = config['nbr_classes']
        hidden_size = config['hidden_size']
        self.dropout = config['dropout']

        if self.graph_module == 'gat':
            self.encoder = GAT(nfeat=nfeat,
                                nhid=hidden_size,
                                nclass=nclass,
                                dropout=self.dropout,
                                nheads=config.get('gat_nhead', 1),
                                alpha=config.get('gat_alpha', 0.2))