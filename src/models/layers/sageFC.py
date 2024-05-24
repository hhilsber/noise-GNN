import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SAGEFC(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, dropout=0.5):
        super().__init__()
        """
        https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py
        """
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = nn.PReLU()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_size, hidden_size))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_size, hidden_size))
        self.convs.append(SAGEConv(hidden_size, out_size))
        self.projection_head = nn.Linear(hidden_size, 128)
        self.bnl = nn.BatchNorm1d(128)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                #h_out = self.act(x)
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        h_out = self.bnl(self.projection_head(x))
        return x, h_out