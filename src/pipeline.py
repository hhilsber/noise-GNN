import torch
import numpy as np
import torch_geometric.utils as tg
from torch.utils.data import DataLoader

from .utils.load_utils import load_network
from .utils.data_utils import sample_neighbor
from .model import NGNN

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

        # Initialize the model
        #self.model = NGNN(config, train_set=datasets.get('train', None))
        self.model = NGNN(config)
        self.model.network = self.model.network.to(self.device)

        # Data loader
        #self.train_loader = dataset
        print("train set = whole dataset for the moment")
        self.train_loader = DataLoader(dataset['features'], batch_size=config['batch_size'], shuffle=True)

        # Config
        self.config = self.model.config

    def train(self):
        print("train")
        for epoch in range(self.config['max_iter']):
            print(' train epoch: {}/{}'.format(epoch+1, self.config['max_iter']))

            self.run_iter(self.train_loader, training=True)

    def run_iter(self, data_loader, training=True):
        self.model.network.train(training)
        network = self.model.network
        
        if training:
            self.model.optimizer.zero_grad()
        
        for batch_idx, data in enumerate(data_loader):
            data = data.to(self.device)
            self.model.optimizer.zero_grad()

            output = self.model.network.encoder(data)
            loss = criterion(output, target)

            pred = torch.round(output)

            # Compute confusion vector between 2 tensors
            confusion_vector = pred / target
            # Compute validation f1 score
            f1_train = compute_metrics(confusion_vector, print_values=False)
            f1_train_history.append(f1_train)
            
            loss.backward()
            optimizer.step()         
            loss_float = loss.item()
            loss_train_history.append(loss_float)