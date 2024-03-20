import torch
import numpy as np
import torch_geometric.utils as tg
#from torch.utils.data import DataLoader

from .utils.load_utils import load_network
from .utils.data_utils import NormalDataset, create_lbl_mat, BCELoss
from .model import NGNN
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.utils import scatter
class Pipeline(object):
    """
    Processing pipeline
    """
    def __init__(self, config):
        # Set metrics:
        #self.train_loss = 
        
        #Set device
        is_cuda = torch.cuda.is_available()
        if not config['no_cuda'] and is_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print('Cuda?  is_available: {} --- version: {} --- device: {}'.format(is_cuda,torch.version.cuda, self.device))
        
        config['device'] = self.device
        
        
        dataset = load_network(config)
        # Data prep
        config['nbr_features'] = dataset['features'].shape[-1]
        config['nbr_classes'] = dataset['labels'].max().item() + 1
        self.train_feat = dataset['features'].float()
        self.train_adj = dataset['adjacency']
        self.lbl_matrix = create_lbl_mat(dataset['labels'])

        # Initialize the model
        #self.model = NGNN(config, train_set=datasets.get('train', None))
        self.model = NGNN(config)
        self.model.network = self.model.network.to(self.device)
        self.criterion = BCELoss(self.lbl_matrix, self.device)

        # Data loader
        print("train set = whole dataset for the moment")
        #train_set = Data(x=dataset['features'], edge_index=dataset['edge_list'])
        #self.train_loader = DataLoader([train_set], batch_size=config['batch_size'], shuffle=False)
        #train_set = NormalDataset(dataset['features'], dataset['adjacency'])
        #self.train_loader = DataLoader(train_set, batch_size=config['nbr_nodes'], shuffle=False)
        
        print("train set = batch size?")

        # Config
        self.config = self.model.config

    def train(self):
        print("train")
        for epoch in range(self.config['max_iter']):
            print(' train epoch: {}/{}'.format(epoch+1, self.config['max_iter']))

            self.run_iter(training=True)

    def run_iter(self, training=True):
        self.model.network.train(training)
        network = self.model.network

        if training:
            self.model.optimizer.zero_grad()
        
        x = self.train_feat.to(self.device)
        adj = self.train_adj.to(self.device)
        
        self.model.optimizer.zero_grad()

        output = self.model.network.encoder(x, adj)
        
        loss = self.criterion(output)
        print(loss)
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