import torch
import torch.optim as optim

from .layers.attention import GAT

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
        #self.network = self.module(self.config)

    def init_optimizer(self):
        """
        parameters = [p for p in self.edge_module.parameters() if p.requires_grad]
        
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        """
        if self.config['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.edge_module.parameters(),
                                                lr=self.config['learning_rate'],
                                                weight_decay=self.config['weight_decay'])