import torch
import torch.optim as optim

from .models.similarity import Similarity

class NGNN(object):
    """
    d
    """
    def __init__(self, config):
        self.config = config
        if self.config['graph_module'] == 'gat':
            self.module = Similarity
        
        self.criterion = None
        self.score_func = None
        self.metric_name = None


        self.init_network()
        self.init_optimizer()

    def init_network(self):
        self.network = self.module(self.config)

    def init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        