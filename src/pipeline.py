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
        
        # Data prep
        self.dataset = load_network(config)
        #print(self.dataset['adjacency'].shape, self.dataset['features'].shape, self.dataset['labels'].shape)
        #print(self.dataset['idx_train'].shape, self.dataset['idx_val'].shape, self.dataset['idx_test'].shape)
        config['nbr_features'] = self.dataset['features'].shape[-1]
        config['nbr_classes'] = self.dataset['labels'].max().item() + 1
        config['nbr_nodes'] = self.dataset['features'].shape[0]
        config['train_size'] = self.dataset['idx_train'].shape[0]

        # Config
        self.config = config
        
        #self.train_feat = dataset['features']
        #self.train_adj = dataset['adjacency']
        #self.lbl_hot = F.one_hot(dataset['labels'], config['nbr_classes']).float()
        #self.lbl_matrix = create_lbl_mat(dataset['labels'])
        
        # Normalized graph laplacian
        #self.norm_GL = normalize_graph_laplacian(self.dataset['adjacency'])
        #self.norm_adj = normalize_adj(self.dataset['adjacency'], config['device'])

        # Initialize the model
        self.device = config['device']
        self.model = NGNN(config)

        self.model.edge_module = self.model.edge_module.to(self.device)
        self.edge_criterion = GRTLoss(config['train_size'], config['alpha'], config['beta'], config['gamma'])
        self.model.network = self.model.network.to(self.device)
        self.network_criterion = nn.CrossEntropyLoss()
        self.reconstruct = Rewire(config['rewire_ratio'], config['device'])
        
        if config['type_train'] == 'dky':
            print('type_train: dont know yet')


    def type_train(self):
        self.run_training()

    def run_training(self, mode='train'):
        if mode == 'train':
            idx = self.dataset['idx_train']
        x = self.dataset['features'][:idx.shape[0],:]
        y = self.dataset['labels'][idx]
        y_hot = F.one_hot(y, self.config['nbr_classes']).float()
        adj = self.dataset['adjacency'][:idx.shape[0],:idx.shape[0]]
        norm_GL = normalize_graph_laplacian(adj)
        norm_adj = normalize_adj_matrix(adj, adj.shape[0], self.device)
        
        model = self.model
        edge_module = self.model.edge_module
        network = self.model.network

        # Epoch
        print('how to rewire?')
        for epoch in range(self.config['max_iter']):
            print(' train epoch: {}/{}'.format(epoch+1, self.config['max_iter']))
            edge_module.train()
            network.train()

            if mode == 'train':
                model.optims.zero_grad()
            
            e_out = self.model.edge_module(x, adj)
            #print(e_out.min(),e_out.max())
            # Rewire
            new_adj = self.reconstruct(e_out, adj)
            new_adj = normalize_adj_matrix(new_adj, new_adj.shape[0], self.device)
            #norm_out = normalize_adj_matrix(e_out, e_out.shape[0], self.device)
            #new_adj = self.config['lambda'] * norm_GL + (1 - self.config['lambda']) * norm_out

            n_out = self.model.network(x, new_adj)

            e_loss = self.edge_criterion(new_adj, x)
            n_loss = self.network_criterion(input=n_out, target=y_hot)
            print(' train loss edge: {}, network {}'.format(e_loss.item(), n_loss.item()))
            loss = e_loss + n_loss

            self.model.optims.zero_grad()
            loss.backward()
            self.model.optims.step()