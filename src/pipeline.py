import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
import matplotlib.pyplot as plt
from ogb.nodeproppred import Evaluator
import datetime as dt

from .utils.load_utils import load_network
from .utils.data_utils import topk_accuracy
from .utils.noise import flip_label
from .models.model import NGNN

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
        self.model1 = NGNN(config)
        self.model2 = NGNN(config)
        self.evaluator = Evaluator(name=config['dataset_name'])

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
  

    def train(self, train_loader, epoch, model1, optimizer1, model2, optimizer2):
        print('Train epoch {}/{}'.format(epoch, self.config['max_epochs']))
        model1.train()
        model2.train()

        pure_ratio_list=[]
        pure_ratio_1_list=[]
        pure_ratio_2_list=[]
        train_total=0
        train_correct=0 
        train_total2=0
        train_correct2=0 

        for batch in train_loader:
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            batch = batch.to(self.device)
            ind = batch.n_id
            # Only consider predictions and labels of seed nodes
            out1 = model1(batch.x, batch.edge_index)[:batch.batch_size]
            out2 = model2(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            yn = batch.yhn[:batch.batch_size].squeeze()
            
            acc1, _ = topk_accuracy(out1, y, batch.batch_size, topk=(1, 5))
            
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
        return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list
    
    def evaluate(self, valid_loader, model1, model2):
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
        model1 = self.model1.network.to(self.device)
        model2 = self.model2.network.to(self.device)
        optimizer1 = self.model1.optimizer
        optimizer2 = self.model2.optimizer

        loss = []
        train_acc_true_hist = []
        train_acc_noise_hist = []
        val_acc_true_hist = []
        val_acc_noise_hist = []
        
        for epoch in range(1, self.config['max_epochs']+1):
            print('adjust lr')
            train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list = self.train(self.train_loader, epoch, model1, optimizer1, model2, optimizer2)
           

            """
            val_acc_true, val_acc_noise = self.evaluate(self.valid_loader, self.model.network)
            val_acc_true_hist.append(val_acc_true)
            val_acc_noise_hist.append(val_acc_noise)"""
            
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
