import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, NeighborLoader
import matplotlib.pyplot as plt
from ogb.nodeproppred import Evaluator
import datetime as dt

from .utils.load_utils import load_network
from .utils.noise import flip_label
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
        self.data.yhn = flip_label(F.one_hot(self.data.y, self.dataset.num_classes).squeeze(), config['flip_rate'])
        
        config['nbr_features'] = self.dataset.num_features #self.dataset.x.shape[-1]
        config['nbr_classes'] = self.dataset.num_classes #dataset.y.max().item() + 1
        config['nbr_nodes'] = self.dataset.x.shape[0]
        
        # Config
        self.config = config

        # Initialize the model
        self.model = NGNN(config)
        self.model.network = self.model.network.to(self.device)
        self.evaluator = Evaluator(name=config['dataset_name'])
        #self.criterion = nn.CrossEntropyLoss()

        print('train: {}, valid: {}, test: {}'.format(self.split_idx['train'].shape[0],self.split_idx['valid'].shape[0],self.split_idx['test'].shape[0]))
        
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
            input_nodes=self.split_idx['valid'],
            num_neighbors=self.config['nbr_neighbors'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            persistent_workers=True
        )
        print(len(self.train_loader),len(self.valid_loader))
        batch = next(iter(self.train_loader))
        print(batch.n_id[:10])
        batch = next(iter(self.valid_loader))
        print(batch.n_id[:10])
        """
        self.valid_loader = NeighborLoader(
            self.data,
            input_nodes=None,
            num_neighbors=[-1],
            batch_size=4096,
            num_workers=self.config['num_workers'],
            persistent_workers=True
        )"""

    def train(self, train_loader, epoch, model1, optimizer1):
        print('Train epoch {}/{}'.format(epoch, self.config['max_epochs']))
        model1.train()

        total_loss = 0
        total_correct_true = 0
        total_correct_noise = 0
        
        for batch in train_loader:
            optimizer1.zero_grad()
            batch = batch.to(self.device)
            out = model1(batch.x, batch.edge_index)

            # Only consider predictions and labels of seed nodes
            out = out[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            yn = batch.yhn[:batch.batch_size].squeeze()
            
            if self.config['noise_loss']:
                loss_1 = F.cross_entropy(out, yn)
            else:
                loss_1 = F.cross_entropy(out, y)
            
            total_loss += float(loss_1)
            total_correct_true += int(out.argmax(dim=-1).eq(y).sum())
            total_correct_noise += int(out.argmax(dim=-1).eq(yn).sum())

            loss_1.backward()
            optimizer1.step()
        train_loss = total_loss / len(train_loader)
        train_acc_true = total_correct_true / self.split_idx['train'].size(0)
        train_acc_noise = total_correct_noise / self.split_idx['train'].size(0)
        return train_loss, train_acc_true, train_acc_noise
    
    def evaluate(self, valid_loader, model1):
        model1.eval()

        total_correct_true = 0
        total_correct_noise = 0
        
        for batch in valid_loader:
            batch = batch.to(self.device)
            out = model1(batch.x, batch.edge_index)
            
            # Only consider predictions and labels of seed nodes
            out = out[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            yn = batch.yhn[:batch.batch_size].squeeze()

            total_correct_true += int(out.argmax(dim=-1).eq(y).sum())
            total_correct_noise += int(out.argmax(dim=-1).eq(yn).sum())
        val_acc_true = total_correct_true / self.split_idx['valid'].size(0)
        val_acc_noise = total_correct_noise / self.split_idx['valid'].size(0)
        return val_acc_true, val_acc_noise

    def loop(self):
        print('loop')
        loss = []
        train_acc_true_hist = []
        train_acc_noise_hist = []
        val_acc_true_hist = []
        val_acc_noise_hist = []
        
        for epoch in range(1, self.config['max_epochs']+1):
            train_loss, train_acc_true, train_acc_noise = self.train(self.train_loader, epoch, self.model.network, self.model.optimizer)
            loss.append(train_loss)
            train_acc_true_hist.append(train_acc_true)
            train_acc_noise_hist.append(train_acc_noise)

            val_acc_true, val_acc_noise = self.evaluate(self.valid_loader, self.model.network)
            val_acc_true_hist.append(val_acc_true)
            val_acc_noise_hist.append(val_acc_noise)
            
        print('train acc true: {:.2f}, train acc noise: {:.2f}, valid acc true: {:.2f}, valid acc noise: {:.2f}'.format(train_acc_true,train_acc_noise,val_acc_true,val_acc_noise))
        if self.config['do_plot']:
            #plt.plot(loss, 'y', label="loss")
            plt.plot(train_acc_true_hist, 'blue', label="train_acc_true")
            plt.plot(val_acc_true_hist, 'red', label="val_acc_true")
            plt.plot(train_acc_noise_hist, 'green', label="train_acc_noise")
            plt.plot(val_acc_noise_hist, 'darkorange', label="val_acc_noise")
            
            plt.legend()
            #plt.show()
            date = dt.datetime.date(dt.datetime.now())
            name = '../plots/dt{}{}_{}_noise_{}_flip{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}.png'.format(date.month,date.day,self.config['module'],self.config['noise_loss'],self.config['flip_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['max_epochs'],self.config['batch_size'],self.config['dropout'])
            plt.savefig(name)

"""
def infer(self, model1, split_idx):
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
"""