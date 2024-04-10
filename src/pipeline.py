import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, NeighborLoader
import matplotlib.pyplot as plt
from ogb.nodeproppred import Evaluator

from .utils.load_utils import load_network
from .utils.data_utils import classification_acc
from .models.model import NGNN

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
class Pipeline(object):
    """
    Processing pipeline
    """
    def __init__(self, config):
        # Set metrics:
        self.device = config['device']

        # Data prep
        self.dataset = load_network(config)
        self.split_idx = self.dataset.get_idx_split()
        self.data = self.dataset[0]

        #config['nbr_features'] = self.dataset.x.shape[-1]
        config['nbr_features'] = self.dataset.num_features
        config['nbr_classes'] = self.dataset.num_classes #dataset.y.max().item() + 1
        config['nbr_nodes'] = self.dataset.x.shape[0]

        # Config
        self.config = config

        # Initialize the model
        self.model = NGNN(config)
        self.model.network = self.model.network.to(self.device)
        self.evaluator = Evaluator(name=config['dataset_name'])
        #self.criterion = nn.CrossEntropyLoss()

        

        self.train_loader = NeighborLoader(
            self.data,
            input_nodes=self.split_idx['train'],
            num_neighbors=self.config['nbr_neighbors'],
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            persistent_workers=True
        )
        
        self.valid_loader = NeighborLoader(
            self.data,
            input_nodes=None,
            num_neighbors=[-1],
            batch_size=4096,
            num_workers=self.config['num_workers'],
            persistent_workers=True
        )

    def train(self, train_loader, epoch, model1, optimizer1):
        print('Train epoch {}/{}'.format(epoch, self.config['max_epochs']))
        model1.train()

        total_loss = 0
        total_correct = 0
        
        for batch in train_loader:
            optimizer1.zero_grad()
            batch = batch.to(self.device)
            out = model1(batch.x, batch.edge_index)

            # Only consider predictions and labels of seed nodes
            out = out[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            
            loss_1 = F.cross_entropy(out, y)
            total_loss += float(loss_1)
            total_correct += int(out.argmax(dim=-1).eq(y).sum())

            loss_1.backward()
            optimizer1.step()
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / self.split_idx['train'].size(0)
        return train_loss, train_acc
    
    def evaluate(self, model1, split_idx):
        model1.eval()

        out = model1.inference(self.data.x, self.valid_loader, self.device)

        y_true = self.data.y.cpu()
        y_pred = out.argmax(dim=-1, keepdim=True)

        train_acc = self.evaluator.eval({
            'y_true': y_true[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        val_acc = self.evaluator.eval({
            'y_true': y_true[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = self.evaluator.eval({
            'y_true': y_true[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })['acc']

        return train_acc, val_acc, test_acc

    def loop(self):
        print('loop')
        loss = []
        train_acc_hist = []
        val_acc_hist = []
        test_acc_hist = []
        for epoch in range(1, self.config['max_epochs']+1):
            train_loss, train_acc = self.train(self.train_loader, epoch, self.model.network, self.model.optimizer)
            loss.append(train_loss)
            #train_acc_hist.append(train_acc)

            train_acc, val_acc, test_acc = self.evaluate(self.model.network, self.split_idx)
            train_acc_hist.append(train_acc)
            val_acc_hist.append(val_acc)
            test_acc_hist.append(test_acc)
        plt.plot(loss, 'g', label="loss")
        plt.plot(train_acc_hist, 'b', label="train_acc")
        plt.plot(val_acc_hist, 'r', label="val_acc")
        plt.plot(test_acc_hist, 'y', label="test_acc")
        plt.legend()
        plt.show()


"""
def evaluate(self, valid_loader, model1):
        model1.eval()
        
        total_loss = 0
        total_correct = 0
        
        for i,batch in enumerate(valid_loader):
            batch = batch.to(self.device)
            out = model1(batch.x, batch.edge_index)
            
            # Only consider predictions and labels of seed nodes
            y = batch.y[:batch.batch_size]
            out = out[:batch.batch_size]

            acc_1 = classification_acc(out, y)
            val_total += 1
            val_acc += acc_1
        val_acc = float(val_acc)/float(val_total) 
        return val_acc"""