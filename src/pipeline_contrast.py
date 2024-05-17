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
from .utils.data_utils import Jensen_Shannon, augment_features, augment_adj
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
        
        # Logger and data loader
        date = dt.datetime.date(dt.datetime.now())
        self.output_name = 'dt{}{}_id{}_{}_{}_{}_noise_{}{}_lay{}_hid{}_lr{}_bs{}_drop{}_epo{}_warmup{}_du{}_tau{}'.format(date.month,date.day,self.config['batch_id'],self.config['train_type'],self.config['algo_type'],self.config['module'],self.config['noise_type'],self.config['noise_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['batch_size'],self.config['dropout'],self.config['max_epochs'],self.config['warmup'],self.config['du'],self.config['tau'])
        self.logger = initialize_logger(self.config, self.output_name)
        np.save('../out_nmat/' + self.output_name + '.npy', noise_mat)

        n = config['nbr_nodes']
        row_indices = np.repeat(np.arange(n), n)
        col_indices = np.tile(np.arange(n), n)
        ones_sparse_matrix = sp.csr_matrix((np.ones(n * n), (row_indices, col_indices)), shape=(n, n))
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
    def warmup(self, epoch, model, optimizer, train_loader):
        model.train()

        total_loss = 0
        total_correct = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out = model(batch.x, batch.edge_index)[0][:batch.batch_size]
            #out = out[:batch.batch_size]
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
        self.logger.info('   Warmup epoch {}/{} --- loss: {:.4f} acc: {:.4f}'.format(epoch+1,self.config['warmup'],train_loss,train_acc))
        
    def jsd(self, model1, model2, train_loader):
        JS_dist = Jensen_Shannon()
        num_samples = self.split_idx['train'].size(0)
        JSD = torch.zeros(num_samples)
        index = torch.zeros(num_samples)

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            with torch.no_grad():
                out1 = torch.nn.Softmax(dim=1)(model1(batch.x, batch.edge_index)[0][:batch.batch_size])
                out2 = torch.nn.Softmax(dim=1)(model2(batch.x, batch.edge_index)[0][:batch.batch_size])
                yhn = batch.yhn[:batch.batch_size].squeeze()

            ## Get the Prediction
            out = (out1 + out2)/2
            dist = JS_dist(out, F.one_hot(yhn, num_classes = self.config['nbr_classes']))
            JSD[int(batch_idx*batch.batch_size):int((batch_idx+1)*batch.batch_size)] = dist
            index[int(batch_idx*batch.batch_size):int((batch_idx+1)*batch.batch_size)] = batch.n_id[:batch.batch_size]
        return JSD, index.int()

    def loop(self):
        print('loop')
        
        self.logger.info('Warmup 1')
        for epoch in range(self.config['warmup']):
            self.warmup(epoch, self.model1.network.to(self.device), self.model1.optimizer, self.train_loader)
        self.logger.info('Warmup 2')
        for epoch in range(self.config['warmup']):
            self.warmup(epoch, self.model2.network.to(self.device), self.model2.optimizer, self.train_loader)
        
        self.logger.info('JSD')
        prob, index = self.jsd(self.model1.network.to(self.device), self.model2.network.to(self.device), self.train_loader)
        
        threshold = torch.mean(prob)
        self.logger.info('thresh1 {:.4f}, du {:.4f}'.format(threshold, self.config['du']))

        if threshold.item() > self.config['du']:
            threshold = threshold - (threshold-torch.min(prob))/self.config['tau']
        self.logger.info('thresh2 {:.4f}, du {:.4f}'.format(threshold, self.config['du']))
        SR = torch.sum(prob < threshold).item()/self.split_idx['train'].size(0)
        self.logger.info('SR {:.4f}'.format(SR))

        bool_tensor = prob < threshold
        
        #self.logger.info('q1 {:.4f}'.format(torch.sum(self.noise_or_not[self.split_idx['train']])))
        #ratio1 = torch.sum(self.noise_or_not[self.split_idx['train']] & ~(bool_tensor)).item()/self.split_idx['train'].size(0)
        #self.logger.info('Warmup ratio2 {:.4f}'.format(ratio2))

        ratio = torch.sum(self.noise_or_not[index]).item()/self.split_idx['train'].size(0)
        s_ratio = torch.sum(self.noise_or_not[index] & (bool_tensor)).item()/self.split_idx['train'].size(0)
        self.logger.info('r {:.4f}'.format(ratio))
        self.logger.info('s {:.4f}'.format(s_ratio))

        pos_index = self.noise_or_not[index] & bool_tensor
        neg_index = ~(self.noise_or_not[index] & bool_tensor)

        pos_ratio = torch.sum(pos_index).item() / self.split_idx['train'].size(0)
        neg_ratio = torch.sum(neg_index).item() / self.split_idx['train'].size(0)

        self.logger.info('pos {:.4f}'.format(pos_ratio))
        self.logger.info('neg {:.4f}'.format(neg_ratio))

        self.logger.info('Done')