import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

"""
import torch_geometric.utils as tg
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.utils import scatter"""

from .utils.load_utils import load_network
from .utils.data_utils import *
from .utils.losses import BCELoss, GRTLoss

from .models.model import NGNN


class Pipeline(object):
    """
    Processing pipeline
    """
    def __init__(self, config):
        # Set metrics:
        
        # Config
        self.config = config
        
        # Data prep
        dataset = load_network(config) #config['nbr_features'] = dataset['features'].shape[-1]    config['nbr_classes'] = dataset['labels'].max().item() + 1
        self.train_feat = dataset['features']
        self.train_adj = dataset['adjacency']
        self.lbl_hot = F.one_hot(dataset['labels'], config['nbr_classes']).float()
        #self.lbl_matrix = create_lbl_mat(dataset['labels'])
        # Normalized graph laplacian
        self.norm_GL = normalize_graph_laplacian(dataset['adjacency'])
        self.norm_adj = normalize_adj(dataset['adjacency'], config['device'])

        # Initialize the model
        self.device = config['device']
        self.model = NGNN(config)

        self.model.edge_module = self.model.edge_module.to(self.device)
        self.edge_criterion = GRTLoss(config['nbr_nodes'], config['alpha'], config['beta'], config['gamma'])
        self.model.network = self.model.network.to(self.device)
        self.network_criterion = nn.CrossEntropyLoss()
        
        if config['type_train'] == 'dky':
            print('type_train: dont know yet')


    def type_train(self):
        self.run_training(training=True)

    def run_training(self, training=True):
        x = self.train_feat.to(self.device)
        adj = self.norm_adj.to(self.device)
        
        # Epoch
        for epoch in range(self.config['max_iter']):
            print(' train epoch: {}/{}'.format(epoch+1, self.config['max_iter']))
            self.model.edge_module.train()
            self.model.network.train()

            if training:
                self.model.optims.zero_grad()
            
            e_out = self.model.edge_module(x, adj)
            # Rewire
            norm_out = normalize_adj(e_out, self.device)
            new_adj = self.config['lambda'] * self.norm_GL + (1 - self.config['lambda']) * norm_out

            n_out = self.model.network(x, adj)

            e_loss = self.edge_criterion(e_out, x)
            n_loss = self.network_criterion(input=n_out, target=self.lbl_hot)
            print(' train loss edge: {}, network {}'.format(e_loss.item(), n_loss.item()))
            loss = e_loss + n_loss

            self.model.optims.zero_grad()
            loss.backward()
            self.model.optims.step()