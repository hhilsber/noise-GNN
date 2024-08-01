import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SAGE(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, dropout=0.5, use_bn=False):
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

        self.use_bn = use_bn
        print('use bn {}'.format(self.use_bn))
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(in_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        if self.use_bn:
            x = self.bn1(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                if self.use_bn:
                    x = self.bn2(x)
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