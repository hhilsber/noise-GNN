import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SAGEPL(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, nbr_nodes, dropout=0.5, use_bn=False):
        super().__init__()
        """
        https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py
        """
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_size, hidden_size))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_size, hidden_size))
        self.convs.append(SAGEConv(hidden_size, out_size))

        # adaptive noise to be added.
        self.noise= nn.Parameter(torch.randn(nbr_nodes, in_size))

        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(in_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, noisy_rate=0.3, n_id=None):

        x_pure, y_pure, z_pure = self.forward_pure(x, edge_index)
        noisy_x = self.adding_noise(x, noisy_rate=noisy_rate, n_id=n_id)
        x_noisy, y_noisy, z_noisy  = self.forward_noisy(noisy_x, edge_index)

        return x_pure, y_pure, z_pure, x_noisy, y_noisy, z_noisy

    def adding_noise(self, x, noisy_rate, n_id=None):
        noisy_x = x.clone()
        if n_id is None:
            noisy_x += torch.sign(noisy_x) * F.normalize(self.noise) * noisy_rate
        else:
            noisy_x += torch.sign(noisy_x) * F.normalize(self.noise[n_id]) * noisy_rate
        return noisy_x

    def forward_pure(self, x, edge_index):
        if self.use_bn:
            x = self.bn1(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                if self.use_bn:
                    x = self.bn2(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                h = x
        return h, torch.log_softmax(x, dim=1), x

    def forward_noisy(self, x, edge_index):
        if self.use_bn:
            x = self.bn1(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                if self.use_bn:
                    x = self.bn2(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                h = x
        return h, torch.log_softmax(x, dim=1), x
    