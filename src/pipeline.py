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
        #self.data.yhn = flip_label(self.data.y, self.dataset.num_classes, config['noise_type'], config['noise_rate'])
        #self.noise_or_not = (self.data.y.squeeze() == self.data.yhn) #.int() # true if same lbl
        
        config['nbr_features'] = self.dataset.num_features #self.dataset.x.shape[-1]
        config['nbr_classes'] = self.dataset.num_classes #dataset.y.max().item() + 1
        config['nbr_nodes'] = self.dataset.x.shape[0]
        
        # Config
        self.config = config

        # Initialize the model
        if self.config['train_type'] in ['coteaching','both']:
            self.model1 = NGNN(config)
            self.model2 = NGNN(config)
            self.criterion = CTLoss(self.device)
            # Drop rate schedule for co-teaching
            self.rate_schedule = np.ones(self.config['max_epochs'])*self.config['noise_rate']*self.config['ct_tau']
            self.rate_schedule[:self.config['ct_tk']] = np.linspace(0, self.config['noise_rate']**self.config['ct_exp'], self.config['ct_tk'])
        if self.config['train_type'] in ['baseline','both']:
            self.model_c = NGNN(config)
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

    def train_ct(self, train_loader, epoch, model1, optimizer1, model2, optimizer2):
        print('Train epoch {}/{}'.format(epoch+1, self.config['max_epochs']))
        model1.train()
        model2.train()

        pure_ratio_1_list=[]
        pure_ratio_2_list=[]
        total_loss_1=0
        total_loss_2=0
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

            total_loss_1 += float(loss_1)
            total_loss_2 += float(loss_2)
            total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())
            total_ratio_1 += (100*pure_ratio_1)
            total_ratio_2 += (100*pure_ratio_2)

            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()
            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()
        train_loss_1 = total_loss_1 / len(train_loader)
        train_loss_2 = total_loss_2 / len(train_loader)
        train_acc_1 = total_correct_1 / self.split_idx['train'].size(0)
        train_acc_2 = total_correct_2 / self.split_idx['train'].size(0)
        pure_ratio_1_list = total_ratio_1 / len(train_loader)
        pure_ratio_2_list = total_ratio_2 / len(train_loader)
        return train_loss_1, train_loss_2, train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list
    
    def train(self, train_loader, epoch, model, optimizer):
        print('Train epoch {}/{}'.format(epoch+1, self.config['max_epochs']))
        model.train()

        total_loss = 0
        total_correct = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            yhn = batch.yhn[:batch.batch_size].squeeze()
            
            loss = F.cross_entropy(out, yhn)
            
            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(y).sum())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / self.split_idx['train'].size(0)
        return train_loss, train_acc

    def evaluate_ct(self, valid_loader, model1, model2):
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

            total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())
        val_acc_1 = total_correct_1 / self.split_idx['valid'].size(0)
        val_acc_2 = total_correct_2 / self.split_idx['valid'].size(0)
        return val_acc_1, val_acc_2
    
    def evaluate(self, valid_loader, model):
        model.eval()

        total_correct = 0
        
        for batch in valid_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()

            total_correct += int(out.argmax(dim=-1).eq(y).sum())
        val_acc = total_correct / self.split_idx['valid'].size(0)
        return val_acc

    def loop(self):
        print('loop')
        print('adjust lr')

        for noise_r in [0.3,0.35,0.4,0.45]:
            self.config['noise_rate'] = noise_r
            self.data.yhn = flip_label(self.data.y, self.dataset.num_classes, self.config['noise_type'], self.config['noise_rate'])
            self.noise_or_not = (self.data.y.squeeze() == self.data.yhn)

            if self.config['train_type'] in ['coteaching','both']:
                print('Train coteaching')
                self.model1.network.reset_parameters()
                self.model2.network.reset_parameters()

                train_loss_1_hist = []
                train_loss_2_hist = []
                train_acc_1_hist = []
                train_acc_2_hist = []
                pure_ratio_1_hist = []
                pure_ratio_2_hist = []
                val_acc_1_hist = []
                val_acc_2_hist = []
                
                for epoch in range(self.config['max_epochs']):
                    train_loss_1, train_loss_2, train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list = self.train_ct(self.train_loader, epoch, self.model1.network.to(self.device), self.model1.optimizer, self.model2.network.to(self.device), self.model2.optimizer)
                    train_loss_1_hist.append(train_loss_1)
                    train_loss_2_hist.append(train_loss_2)
                    train_acc_1_hist.append(train_acc_1)
                    train_acc_2_hist.append(train_acc_2)
                    pure_ratio_1_hist.append(pure_ratio_1_list)
                    pure_ratio_2_hist.append(pure_ratio_2_list)

                    val_acc_1, val_acc_2 = self.evaluate_ct(self.valid_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
                    val_acc_1_hist.append(val_acc_1)
                    val_acc_2_hist.append(val_acc_2)
            
            if self.config['train_type'] in ['baseline','both']:
                print('Train baseline')
                self.model_c.network.reset_parameters()

                train_loss_hist = []
                train_acc_hist = []
                val_acc_hist = []
                for epoch in range(self.config['max_epochs']):
                    train_loss, train_acc = self.train(self.train_loader, epoch, self.model_c.network.to(self.device), self.model_c.optimizer)
                    train_loss_hist.append(train_loss)
                    train_acc_hist.append(train_acc)

                    val_acc = self.evaluate(self.valid_loader, self.model_c.network.to(self.device))
                    val_acc_hist.append(val_acc)
            
            if self.config['do_plot']:
                fig, axs = plt.subplots(3, 1, figsize=(10, 15))
                
                axs[0].axhline(y=0.8, color='grey', linestyle='--')
                axs[0].axhline(y=0.9, color='grey', linestyle='--')
                if self.config['train_type'] in ['coteaching','both']:
                    line1, = axs[0].plot(train_acc_1_hist, 'blue', label="train_acc_1_hist")
                    line2, = axs[0].plot(train_acc_2_hist, 'darkgreen', label="train_acc_2_hist")
                    line3, = axs[0].plot(val_acc_1_hist, 'purple', label="val_acc_1_hist")
                    line4, = axs[0].plot(val_acc_2_hist, 'chartreuse', label="val_acc_2_hist")
                    
                    axs[1].plot(pure_ratio_1_hist, 'blue', label="pure_ratio_1_hist")
                    axs[1].plot(pure_ratio_2_hist, 'darkgreen', label="pure_ratio_2_hist")
                    axs[1].legend()

                    axs[2].plot(train_loss_1_hist, 'blue', label="train_loss_1_hist")
                    axs[2].plot(train_loss_2_hist, 'darkgreen', label="train_loss_2_hist")
                    
                if self.config['train_type'] in ['baseline','both']:
                    line5, = axs[0].plot(train_acc_hist, 'red', label="train_acc_hist")
                    line6, = axs[0].plot(val_acc_hist, 'peachpuff', label="val_acc_hist")

                    axs[2].plot(train_loss_hist, 'red', label="train_loss_hist")
                
                if self.config['train_type'] in ['coteaching']:
                    axs[0].legend(handles=[line1, line2, line3, line4], loc='upper left', bbox_to_anchor=(1.05, 1))
                elif self.config['train_type'] in ['baseline']:
                    axs[0].legend(handles=[line6, line5], loc='upper left', bbox_to_anchor=(1.05, 1))
                else:
                    axs[0].legend(handles=[line1, line2, line3, line4, line5, line6], loc='upper left', bbox_to_anchor=(1.05, 1))
                
                axs[0].set_title('Plot 1')
                axs[1].set_title('Plot 2')
                axs[2].legend()
                axs[2].set_title('Plot 3')

                plt.tight_layout()
                #plt.show()
                date = dt.datetime.date(dt.datetime.now())
                name = '../plots/coteaching/dt{}{}_{}_{}_noise_{}{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_ctck{}_ctexp{}_cttau{}_neigh{}{}{}.png'.format(date.month,date.day,self.config['train_type'],self.config['module'],self.config['noise_type'],self.config['noise_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['max_epochs'],self.config['batch_size'],self.config['dropout'],self.config['ct_tk'],self.config['ct_exp'],self.config['ct_tau'],self.config['nbr_neighbors'][0],self.config['nbr_neighbors'][1],self.config['nbr_neighbors'][2])
                plt.savefig(name)
