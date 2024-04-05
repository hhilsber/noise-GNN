import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, NeighborLoader
import matplotlib.pyplot as plt

from .utils.load_utils import load_network
from .utils.data_utils import classification_acc
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
        #self.criterion = nn.CrossEntropyLoss()

        self.train_loader = NeighborLoader(
            self.dataset[0],
            num_neighbors=[self.config['nbr_neighbors']] * self.config['k_hops'],
            batch_size=self.config['batch_size'],
            input_nodes=self.dataset.train_idx,
            is_sorted=False,
            shuffle=False
        )
        
        self.valid_loader = NeighborLoader(
            self.dataset[0],
            num_neighbors=[self.config['nbr_neighbors']] * self.config['k_hops'],
            batch_size=self.config['batch_size'],
            input_nodes=self.dataset.valid_idx,
            is_sorted=False,
            shuffle=False
        )

    def train(self, train_loader, epoch, model1, optimizer1):
        print('Train epoch {}/{}'.format(epoch, self.config['max_epochs']))
        model1.train()

        train_total=0
        train_loss=0
        train_acc=0 
        #for batch in train_loader:
        for i,batch in enumerate(train_loader):
            batch = batch.to(self.device)
            out = model1(batch.x, batch.edge_index)
            #out = model1(batch)

            # Only consider predictions and labels of seed nodes
            y = batch.y[:batch.batch_size]
            out = out[:batch.batch_size]
            #out = torch.max(out, dim=1)[1]
            
            #loss_1 = self.criterion(input=out, target=y.squeeze())
            loss_1 = F.cross_entropy(out, y.squeeze())
            acc_1 = classification_acc(out, y)
            
            train_total += 1
            train_loss += loss_1.item()
            train_acc += acc_1

            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()
        train_loss = float(train_loss)/float(train_total)
        train_acc = float(train_acc)/float(train_total)
        return train_loss, train_acc #torch.tensor([0.])
    
    def evaluate(self, valid_loader, model1):
        model1.eval()
        
        val_total=0
        val_acc=0 
        for i,batch in enumerate(valid_loader):
            batch = batch.to(self.device)
            out = model1(batch.x, batch.edge_index)
            #out = model1(batch)
            # Only consider predictions and labels of seed nodes
            y = batch.y[:batch.batch_size]
            out = out[:batch.batch_size]

            acc_1 = classification_acc(out, y)
            val_total += 1
            val_acc += acc_1
        val_acc = float(val_acc)/float(val_total) 
        return val_acc

    def loop(self):
        print('loop')
        loss = []
        train_acc_hist = []
        val_acc_hist = []
        for epoch in range(1, self.config['max_epochs']+1):
            train_loss, train_acc = self.train(self.train_loader, epoch, self.model.network, self.model.optimizer)
            loss.append(train_loss)
            train_acc_hist.append(train_acc)

            val_acc = self.evaluate(self.valid_loader, self.model.network)
            val_acc_hist.append(val_acc)
        plt.plot(loss, 'g', label="loss")
        plt.plot(train_acc_hist, 'b', label="train_acc")
        plt.plot(val_acc_hist, 'r', label="val_acc")
        plt.legend()
        plt.show()