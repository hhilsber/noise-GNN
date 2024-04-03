import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class SimpleGCN(nn.Module):
    def __init__(self, in_size, hidden_dim, out_size, dropout_prob=0.5):
        super(SimpleGCN, self).__init__()
        
        self.conv1 = GCNConv(in_size, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First graph convolutional layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second graph convolutional layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Global pooling to aggregate node embeddings
        x = global_mean_pool(x, data.batch)
        
        # Log-softmax activation for output
        x = F.log_softmax(x, dim=1)
        
        return x







#####################################################################################
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
class GraphConvolution(Module):

    #Graph convolution layer, ref https://github.com/tkipf/pygcn


    def __init__(self, in_size, out_size, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = Parameter(torch.FloatTensor(in_size, out_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)"""