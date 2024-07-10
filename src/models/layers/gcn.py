import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv



class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, out_channels, normalize=False))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge)
        return x # x.log_softmax(dim=-1)


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