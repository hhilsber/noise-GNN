import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, out_channels, normalize=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, training=True):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=training)
            x = conv(x, edge_index)
        return x #x.softmax(dim=-1)


"""
class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_size, out_size))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
        return x #torch.log_softmax(x, dim=-1)
        """