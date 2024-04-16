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
from .utils.losses import CTLoss

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
        self.data.yhn = flip_label(F.one_hot(self.data.y, self.dataset.num_classes).squeeze(), config['noise_rate'])
        self.noise_or_not = (self.data.y.squeeze() == self.data.yhn) #.int() # true if same lbl
        
        config['nbr_features'] = self.dataset.num_features #self.dataset.x.shape[-1]
        config['nbr_classes'] = self.dataset.num_classes #dataset.y.max().item() + 1
        config['nbr_nodes'] = self.dataset.x.shape[0]
        
        """
        # Drop rate schedule for co-teaching
        self.rate_schedule = np.ones(config['max_epochs'])*config['noise_rate']
        self.rate_schedule[:config['ct_tk']] = np.linspace(0, config['noise_rate']**config['ct_exp'], config['ct_tk'])
        print(self.rate_schedule)"""
        # Config
        self.config = config

        # Initialize the model
        self.model1 = NGNN(config)
        self.model2 = NGNN(config)
        self.evaluator = Evaluator(name=config['dataset_name'])
        self.criterion = CTLoss(self.device)

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
        print('Train epoch {}/{}'.format(epoch+1, self.config['max_epochs']))
        model1.train()
        model2.train()

        pure_ratio_1_list=[]
        pure_ratio_2_list=[]
        total_correct_1=0
        total_correct_2=0
        total_ratio_1=0
        total_ratio_2=0

        for batch in train_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out1 = model1(batch.x, batch.edge_index)[:batch.batch_size]
            out2 = model2(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            yhn = batch.yhn[:batch.batch_size].squeeze()
            
            loss_1, loss_2, pure_ratio_1, pure_ratio_2 = self.criterion(out1, out2, yhn, self.rate_schedule[epoch], batch.n_id, self.noise_or_not)
            total_ratio_1 += pure_ratio_1
            total_ratio_2 += pure_ratio_2

            total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())

            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()
            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()
        
        train_acc_1 = total_correct_1 / self.split_idx['train'].size(0)
        train_acc_2 = total_correct_2 / self.split_idx['train'].size(0)
        pure_ratio_1_list = total_ratio_1 / self.split_idx['train'].size(0)
        pure_ratio_2_list = total_ratio_2 / self.split_idx['train'].size(0)
        return train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list
    
    def evaluate(self, valid_loader, model1, model2):
        model1.eval()
        model2.eval()

        total_correct_1 = 0
        total_correct_2 = 0
        
        for batch in valid_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out1 = model1(batch.x, batch.edge_index)[:batch.batch_size]
            out2 = model2(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            yn = batch.yhn[:batch.batch_size].squeeze()

            total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())
        val_acc_1 = total_correct_1 / self.split_idx['valid'].size(0)
        val_acc_2 = total_correct_2 / self.split_idx['valid'].size(0)
        return val_acc_1, val_acc_2

    def loop(self):
        print('loop')
        model1 = self.model1.network.to(self.device)
        model2 = self.model2.network.to(self.device)
        optimizer1 = self.model1.optimizer
        optimizer2 = self.model2.optimizer
        print('adjust lr')

        for cttk in [4, 8]:
            for ctexp in [0.5, 2]:
                self.config['ct_tk'] = cttk
                self.config['ct_exp'] = ctexp
                print('----- cttk: {}, ctexp: {}'.format(self.config['ct_tk'], self.config['ct_exp']))
                
                # Drop rate schedule for co-teaching
                self.rate_schedule = np.ones(self.config['max_epochs'])*self.config['noise_rate']
                self.rate_schedule[:self.config['ct_tk']] = np.linspace(0, self.config['noise_rate']**self.config['ct_exp'], self.config['ct_tk'])
                print(self.rate_schedule)

                model1.reset_parameters()
                model2.reset_parameters()
                train_acc_1_hist = []
                train_acc_2_hist = []
                pure_ratio_1_hist = []
                pure_ratio_2_hist = []
                val_acc_1_hist = []
                val_acc_2_hist = []
                
                
                for epoch in range(self.config['max_epochs']):
                    train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list = self.train(self.train_loader, epoch, model1, optimizer1, model2, optimizer2)
                    train_acc_1_hist.append(train_acc_1)
                    train_acc_2_hist.append(train_acc_2)
                    pure_ratio_1_hist.append(pure_ratio_1_list)
                    pure_ratio_2_hist.append(pure_ratio_2_list)

                    val_acc_1, val_acc_2 = self.evaluate(self.valid_loader, model1, model2)
                    val_acc_1_hist.append(val_acc_1)
                    val_acc_2_hist.append(val_acc_2)
                    
                print('train_acc_1: {:.2f}, train_acc_2: {:.2f}, val_acc_1: {:.2f}, val_acc_2: {:.2f}'.format(train_acc_1,train_acc_2,val_acc_1,val_acc_2))
                if self.config['do_plot']:
                    plt.figure()
                    plt.subplot(211)
                    plt.plot(train_acc_1_hist, 'blue', label="train_acc_1_hist")
                    plt.plot(train_acc_2_hist, 'red', label="train_acc_2_hist")
                    plt.plot(val_acc_1_hist, 'green', label="val_acc_1_hist")
                    plt.plot(val_acc_2_hist, 'darkorange', label="val_acc_2_hist")
                    plt.ylim(0,1)
                    plt.legend()
                    plt.subplot(212)
                    plt.plot(pure_ratio_1_hist, 'blue', label="pure_ratio_1_hist")
                    plt.plot(pure_ratio_2_hist, 'red', label="pure_ratio_2_hist")
                    plt.legend()
                    #plt.show()
                    date = dt.datetime.date(dt.datetime.now())
                    name = '../plots/coteaching/dt{}{}_{}_noise{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_ctck{}_ctexp{}.png'.format(date.month,date.day,self.config['module'],self.config['noise_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['max_epochs'],self.config['batch_size'],self.config['dropout'],self.config['ct_tk'],self.config['ct_exp'])
                    plt.savefig(name)
