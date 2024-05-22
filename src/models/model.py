import torch
import torch.optim as optim

from .layers.convolution import SimpleGCN
from .layers.sage import SAGE
from .layers.sageFC import SAGEFC


class NGNN(object):
    """
    d
    """
    def __init__(self, in_size=100, hidden_size=128, out_size=47, num_layers=2, dropout=0.5, lr=0.001, optimizer='adam', module='sageFC'):
        #self.config = config
        
        self.criterion = None
        self.score_func = None
        self.metric_name = None

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lr = lr
        self.optimizer = optimizer
        self.module = module

        self.init_network()
        self.init_optimizer()

    def init_network(self):
        if self.module == 'gcn':
            self.network = SimpleGCN(in_size=self.in_size,
                                hidden_size=self.hidden_size,
                                out_size=self.out_size,
                                num_layers=self.num_layers,
                                dropout=self.dropout)
        elif self.module == 'sage':
            self.network = SAGE(in_size=self.in_size,
                                    hidden_size=self.hidden_size,
                                    out_size=self.out_size,
                                    num_layers=self.num_layers,
                                    dropout=self.dropout)
        elif self.module == 'sageFC':
            self.network = SAGEFC(in_size=self.in_size,
                                    hidden_size=self.hidden_size,
                                    out_size=self.out_size,
                                    num_layers=self.num_layers,
                                    dropout=self.dropout)


    def init_optimizer(self):
        if self.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(),
                                            lr=self.lr)
        elif self.optimizer == 'single_adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(),
                                                lr=self.lr,
                                                weight_decay=self.weight_decay)
        elif self.optimizer == 'double_adam':
            self.optims = MultipleOptimizer(torch.optim.Adam(self.edge_module.parameters(),
                                                lr=self.lr,
                                                weight_decay=self.weight_decay),
                                            torch.optim.Adam(self.network.parameters(),
                                                lr=self.lr,
                                                weight_decay=self.weight_decay))


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