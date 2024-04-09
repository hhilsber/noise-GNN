import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        """
        https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py
        """
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def inference(self, x_all):
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