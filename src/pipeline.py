import torch
import numpy as np
import torch_geometric.utils as tg
"""
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.utils import scatter"""

from .utils.load_utils import load_network
from .utils.data_utils import NormalDataset, create_lbl_mat, BCELoss
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
        self.train_feat = dataset['features'].float()
        self.train_adj = dataset['adjacency']
        self.lbl_matrix = create_lbl_mat(dataset['labels'])

        # Initialize the model
        self.device = config['device']
        self.model = NGNN(config)

        self.model.edge_module = self.model.edge_module.to(self.device)
        self.criterion = BCELoss(self.lbl_matrix, self.device)
        
        if config['type_train'] == 'dky':
            print('type_train: dont know yet')

        # Data loader
        print("      train set = whole dataset for the moment")
        #train_set = Data(x=dataset['features'], edge_index=dataset['edge_list'])
        #self.train_loader = DataLoader([train_set], batch_size=config['batch_size'], shuffle=False)
        #train_set = NormalDataset(dataset['features'], dataset['adjacency'])
        #self.train_loader = DataLoader(train_set, batch_size=config['nbr_nodes'], shuffle=False)
        
        print("       train set = batch size?")

    def type_train(self):
        print("train")
        self.run_training(training=True)

    def run_training(self, training=True):
        for epoch in range(self.config['max_iter']):
            print(' train epoch: {}/{}'.format(epoch+1, self.config['max_iter']))
            self.model.edge_module.train()

            if training:
                self.model.optimizer.zero_grad()
            
            x = self.train_feat.to(self.device)
            adj = self.train_adj.to(self.device)
            
            self.model.optimizer.zero_grad()

            output = self.model.edge_module(x, adj)
            
            loss = self.criterion(output)
            loss.backward()
            self.model.optimizer.step()      
            """
            pred = torch.round(output)

            # Compute confusion vector between 2 tensors
            confusion_vector = pred / target
            # Compute validation f1 score
            f1_train = compute_metrics(confusion_vector, print_values=False)
            f1_train_history.append(f1_train)
            
            loss.backward()
            optimizer.step()         
            loss_float = loss.item()
            loss_train_history.append(loss_float)"""