import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from .utils.load_utils import load_network
from .models.model import NGNN

class Pipeline(object):
    """
    Processing pipeline
    """
    def __init__(self, config):
        # Set metrics:
        
        # Data prep
        self.dataset = load_network(config)
        
        config['nbr_features'] = self.dataset.x.shape[-1]
        config['nbr_classes'] = self.dataset.y.max().item() + 1
        config['nbr_nodes'] = self.dataset.x.shape[0]
        config['train_size'] = self.dataset.train_idx.shape[0]
        
        # Config
        self.config = config
        
        # Initialize the model
        self.device = config['device']
        self.model = NGNN(config)
        
        self.model.network = self.model.network.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.train_loader = NeighborLoader(
            self.dataset[0],
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[self.config['nbr_neighbors']] * self.config['k_hops'],
            # Use a batch size of 128 for sampling training nodes
            batch_size=self.config['batch_size'],
            input_nodes=self.dataset.train_idx,
        )

    def train(self, train_loader, epoch, model1, optimizer1):
        print('Train epoch {}/{}'.format(epoch, self.config['max_epochs']))
        
        for batch in train_loader:
            batch = batch.to(self.device)
            #out = model(batch.x, batch.edge_index)
            
        return torch.tensor([0.])

    def loop(self):
        print('loop')

        for epoch in range(1, self.config['max_epochs']+1):
            self.model.network.train()
            
            out = self.train(self.train_loader, epoch, self.model.network, self.model.optimizer)
        