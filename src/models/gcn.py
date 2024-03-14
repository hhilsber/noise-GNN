import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np

class GCN(nn.Module):
    """
    graph conv 
    """

    def __init__(self, config):

        super(GCN, self).__init__()

        self.device = config['device']
        self.nfeat = config['num_feat']
        self.nclass = config['num_class']
        self.hidden_size = config['hidden_size']
        
        self.gc1 = GCNConv(config['num_feat'], config['hidden_size'], bias=with_bias,add_self_loops=self_loop)
        self.gc2 = GCNConv(config['hidden_size'], config['num_class'], bias=with_bias,add_self_loops=self_loop)
        self.dropout = config['dropout']
        self.lr = config['learning_rate']

        self.optim = config['optimizer']
        self.iter = config['max_iter']
        
        # weight decay ?
        
    def forward(self, x, edge_index, edge_weight):
        if self.with_relu:
            x = F.relu(self.gc1(x, edge_index,edge_weight))
        else:
            x = self.gc1(x, edge_index,edge_weight)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index,edge_weight)
        return x