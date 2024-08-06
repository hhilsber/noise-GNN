import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class SimpleGCN(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, dropout=0.5, use_bn=False):
        super(SimpleGCN, self).__init__()
        
        """
        https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/gnn.py

        """
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_size, hidden_size, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_size, hidden_size, normalize=False))
        self.convs.append(GCNConv(hidden_size, out_size, normalize=False))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to(device)
                edge_index = batch.edge_index.to(device)
                x = self.convs[i](x, edge_index)
                x = x[:batch.batch_size]
                if i != self.num_layers - 1:
                    x = x.relu()
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)
        return x_all






#####################################################################################

"""
def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
"""

###

"""

class SimpleGCN(nn.Module):
    def __init__(self, in_size, hidden_dim, out_size, dropout_prob=0.3):
        super(SimpleGCN, self).__init__()
        
        self.conv1 = GCNConv(in_size, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_size)
        #self.dropout = nn.Dropout(dropout_prob)
        self.dropout = dropout_prob
        self.batch_norm = torch.nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, training=True):
        #x, edge_index = data.x, data.edge_index

        # Conv1
        x = self.conv1(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = F.dropout(x, p=self.dropout, training=training)

        
        # Conv2
        x = self.conv2(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)

        # Global pooling to aggregate node embeddings
        #x = global_mean_pool(x, data.batch)

        # Conv3
        x = self.conv3(x, edge_index)
        
        # Log-softmax activation for output
        x = F.log_softmax(x, dim=1)
        
        return x









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