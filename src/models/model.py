import torch
import torch.optim as optim

from .layers.attention import GAT
from .layers.convolution import GCN

class NGNN(object):
    """
    d
    """
    def __init__(self, config):
        self.config = config
        
        self.criterion = None
        self.score_func = None
        self.metric_name = None


        self.init_network()
        self.init_optimizer()

    def init_network(self):
        if self.config['graph_edge_module'] == 'gat':
            self.edge_module = GAT(nnode=self.config['nbr_nodes'],
                                nfeat=self.config['nbr_features'],
                                nclass=self.config['nbr_classes'],
                                nhid=self.config['hidden_size'],
                                dropout=self.config['dropout'],
                                nheads=self.config['gat_nhead'],
                                alpha=self.config['gat_alpha'])
            self.network = GCN(nfeat=self.config['nbr_features'],
                                nclass=self.config['nbr_classes'],
                                nhid=self.config['hidden_size'],
                                dropout=self.config['dropout'])

    def init_optimizer(self):
        
        if self.config['optimizer'] == 'adam':
            """
            self.edge_optimizer = torch.optim.Adam(self.edge_module.parameters(),
                                                lr=self.config['learning_rate'],
                                                weight_decay=self.config['weight_decay'])
            self.network_optimizer = torch.optim.Adam(self.network.parameters(),
                                                lr=self.config['learning_rate'],
                                                weight_decay=self.config['weight_decay'])"""
            self.optims = MultipleOptimizer(torch.optim.Adam(self.edge_module.parameters(),
                                                lr=self.config['learning_rate'],
                                                weight_decay=self.config['weight_decay']),
                                            torch.optim.Adam(self.network.parameters(),
                                                lr=self.config['learning_rate'],
                                                weight_decay=self.config['weight_decay']))


class MultipleOptimizer():
    """ a class that wraps multiple optimizers """
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def update_lr(self, op_index, new_lr):
        """ update the learning rate of one optimizer
        Parameters: op_index: the index of the optimizer to update
                    new_lr:   new learning rate for that optimizer """
        for param_group in self.optimizers[op_index].param_groups:
            param_group['lr'] = new_lr