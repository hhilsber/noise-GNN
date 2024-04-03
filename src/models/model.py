import torch
import torch.optim as optim

from .layers.attention import GAT, SIMA
from .layers.convolution import SimpleGCN

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
        if self.config['module'] == 'simple_gcn':
            self.network = SimpleGCN(in_size=self.config['nbr_features'],
                                hidden_dim=self.config['hidden_size'],
                                out_size=self.config['nbr_classes'],
                                dropout_prob=self.config['dropout'])


    def init_optimizer(self):
        if self.config['optimizer'] == 'single_adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(),
                                                lr=self.config['learning_rate'],
                                                weight_decay=self.config['weight_decay'])

        elif self.config['optimizer'] == 'double_adam':
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