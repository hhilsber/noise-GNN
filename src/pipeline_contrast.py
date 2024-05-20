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
from .utils.data_utils import Jensen_Shannon
from .utils.augmentation import augment_edges_pos, augment_edges_neg, shuffle_pos, shuffle_neg
from .utils.utils import initialize_logger
from .utils.noise import flip_label
from .models.model import NGNN
from .utils.losses import *

class PipelineCT(object):
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
        self.data.yhn, noise_mat = flip_label(self.data.y, self.dataset.num_classes, config['noise_type'], config['noise_rate'])
        self.noise_or_not = (self.data.y.squeeze() == self.data.yhn) #.int() # true if same lbl
        
        config['nbr_features'] = self.dataset.num_features
        config['nbr_classes'] = self.dataset.num_classes
        config['nbr_nodes'] = self.dataset.x.shape[0]

        # Config
        self.config = config

        # Initialize the model
        self.model1 = NGNN(config)
        self.model2 = NGNN(config)
        self.evaluator = Evaluator(name=config['dataset_name'])
        # Coteaching param
        self.criterion = CTLoss(self.device)
        self.rate_schedule = np.ones(self.config['warmup'])*self.config['noise_rate']*self.config['ct_tau']
        self.rate_schedule[:self.config['ct_tk']] = np.linspace(0, self.config['noise_rate'], self.config['ct_tk'])
        print('rate_schedule: {}'.format(self.rate_schedule))

        # Logger and data loader
        date = dt.datetime.date(dt.datetime.now())
        self.output_name = 'dt{}{}_id{}_{}_{}_{}_noise_{}{}_lay{}_hid{}_lr{}_bs{}_drop{}_epo{}_warmup{}_cttk{}_cttau{}'.format(date.month,date.day,self.config['batch_id'],self.config['train_type'],self.config['algo_type'],self.config['module'],self.config['noise_type'],self.config['noise_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['batch_size'],self.config['dropout'],self.config['max_epochs'],self.config['warmup'],self.config['ct_tk'],self.config['ct_tau'])
        self.logger = initialize_logger(self.config, self.output_name)
        np.save('../out_nmat/' + self.output_name + '.npy', noise_mat)

        # Graph augmentation
        if config['augment']:
            edge_pos = augment_edges_pos(self.data.edge_index, config['nbr_nodes'], config['edge_prob'])
            edge_neg = augment_edges_neg(self.data.edge_index, config['nbr_nodes'])
            features_shuffled_pos = shuffle_pos(self.data.x, config['device'], config['feat_prob'])
            features_shuffled_neg = shuffle_neg(self.data.x, config['device'])
        print('ok')

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
    
    def warmup(self, train_loader, epoch, model1, optimizer1, model2, optimizer2):
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
            out1 = model1(batch.x, batch.edge_index)[0][:batch.batch_size]
            out2 = model2(batch.x, batch.edge_index)[0][:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            yhn = batch.yhn[:batch.batch_size].squeeze()
            
            loss_1, loss_2, pure_ratio_1, pure_ratio_2, _, _, _, _ = self.criterion(out1, out2, yhn, self.rate_schedule[epoch], batch.n_id, self.noise_or_not)
            
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
        self.logger.info('   Warmup epoch {}/{} --- loss1: {:.3f} loss2: {:.3f} acc1: {:.3f} acc2: {:.3f}'.format(epoch+1,self.config['warmup'],train_loss_1,train_loss_2,train_acc_1,train_acc_2))
        #return train_loss_1, train_loss_2, train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list

    def split(self, epoch, model1, model2, train_loader):
        # Pred
        clean_1 = torch.tensor([])
        clean_2 = torch.tensor([])
        noisy_1 = torch.tensor([])
        noisy_2 = torch.tensor([])
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            with torch.no_grad():
                out1 = torch.nn.Softmax(dim=1)(model1(batch.x, batch.edge_index)[0][:batch.batch_size])
                out2 = torch.nn.Softmax(dim=1)(model2(batch.x, batch.edge_index)[0][:batch.batch_size])
                yhn = batch.yhn[:batch.batch_size].squeeze()

                _, _, _, _, ind_clean_1, ind_clean_2, ind_noisy_1, ind_noisy_2 = self.criterion(out1, out2, yhn, self.rate_schedule[epoch], batch.n_id, self.noise_or_not)
            
            clean_1 = torch.cat((clean_1, ind_clean_1), dim=0)
            clean_2 = torch.cat((clean_2, ind_clean_2), dim=0)
            noisy_1 = torch.cat((noisy_1, ind_noisy_1), dim=0)
            noisy_2 = torch.cat((noisy_2, ind_noisy_2), dim=0)
        return clean_1.long(), clean_2.long(), noisy_1.long(), noisy_2.long()

    def loop(self):
        print('loop')
        
        # Warmup
        self.logger.info('Warmup')
        for epoch in range(self.config['warmup']):
            self.warmup(self.train_loader, epoch, self.model1.network.to(self.device), self.model1.optimizer, self.model2.network.to(self.device), self.model2.optimizer)
        
        # Split data in clean and noisy sets
        self.logger.info('Split epoch {}'.format(epoch))
        clean_1, clean_2, noisy_1, noisy_2 = self.split(epoch, self.model1.network.to(self.device), self.model2.network.to(self.device), self.train_loader)

        # Check stats
        clean_ratio1 = torch.sum(self.noise_or_not[clean_1]).item()/self.split_idx['train'].size(0)
        clean_ratio2 = torch.sum(self.noise_or_not[clean_2]).item()/self.split_idx['train'].size(0)
        self.logger.info('clean ratio1 {:.4f}'.format(clean_ratio1))
        self.logger.info('clean ratio1 {:.4f}'.format(clean_ratio2))
        noisy_ratio1 = torch.sum(self.noise_or_not[noisy_1]).item()/self.split_idx['train'].size(0)
        noisy_ratio2 = torch.sum(self.noise_or_not[noisy_2]).item()/self.split_idx['train'].size(0)
        self.logger.info('noisy ratio1 {:.4f}'.format(noisy_ratio1))
        self.logger.info('noisy ratio2 {:.4f}'.format(noisy_ratio2))


        self.logger.info('Done')