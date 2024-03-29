import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


from .utils.load_utils import load_network
from .utils.data_utils import *
from .utils.losses import BCELoss, GRTLoss

from .models.model import NGNN


class Pipeline(object):
    """
    Processing pipeline
    """
    def __init__(self, config):
        # Set metrics:
        
        # Data prep
        self.dataset = load_network(config)
        #print(self.dataset['adjacency'].shape, self.dataset['features'].shape, self.dataset['labels'].shape)
        #print(self.dataset['idx_train'].shape, self.dataset['idx_val'].shape, self.dataset['idx_test'].shape)
        config['nbr_features'] = self.dataset['features'].shape[-1]
        config['nbr_classes'] = self.dataset['labels'].max().item() + 1
        config['nbr_nodes'] = self.dataset['features'].shape[0]
        config['train_size'] = self.dataset['idx_train'].shape[0]

        # Config
        self.config = config
        
        #self.train_feat = dataset['features']
        #self.train_adj = dataset['adjacency']
        #self.lbl_hot = F.one_hot(dataset['labels'], config['nbr_classes']).float()
        #self.lbl_matrix = create_lbl_mat(dataset['labels'])
        
        # Normalized graph laplacian
        #self.norm_GL = normalize_graph_laplacian(self.dataset['adjacency'])
        #self.norm_adj = normalize_adj(self.dataset['adjacency'], config['device'])

        # Initialize the model
        self.device = config['device']
        self.model = NGNN(config)

        self.model.edge_module = self.model.edge_module.to(self.device)
        self.edge_criterion = GRTLoss(config['train_size'], config['alpha'], config['beta'], config['gamma'], config['device'])
        self.model.network = self.model.network.to(self.device)
        self.network_criterion = nn.CrossEntropyLoss()
        self.reconstruct = Rewire(config['rewire_ratio'], config['device'])
        
        if config['type_train'] == 'dky':
            print('type_train: dont know yet')


    def type_train(self):
        self.run_training()

    def run_training(self, mode='train'):
        idx_train = self.dataset['idx_train']
        idx_val = self.dataset['idx_val']

        x = self.dataset['features'][idx_train]
        y = self.dataset['labels'][idx_train]
        print(y[:10])
        y_hot = F.one_hot(y, self.config['nbr_classes']).float()
        adj = self.dataset['adjacency'][:idx_train.shape[0],:idx_train.shape[0]]
        print(adj[:10,:10])
        
        norm_GL = normalize_graph_laplacian(adj)
        norm_adj = normalize_adj_matrix(adj, adj.shape[0], self.device)
        
        model = self.model
        edge_module = self.model.edge_module
        network = self.model.network

        loss_edge = []
        sm = []
        con = []
        spar = []
        loss_pred = []
        loss_total = []
        # Epoch
        print('how to rewire?')
        for epoch in range(self.config['max_iter']):
            print(' train epoch: {}/{}'.format(epoch+1, self.config['max_iter']))
            #print('nan 0: {}'.format(torch.count_nonzero(torch.isnan(norm_adj))))
            edge_module.train()
            #network.train()
            
            if mode == 'train':
                model.optims.zero_grad()
            
            e_out = edge_module(x)
            #print('nan 1: {}'.format(torch.count_nonzero(torch.isnan(e_out))))
            #print(e_out.min(),e_out.max())
            # Rewire$
            
            e_rec = self.reconstruct(e_out, norm_adj)
            
            #print('nan 2: {}'.format(torch.count_nonzero(torch.isnan(e_out))))
            #print(new_adj[:10,:10])
            e_rec = normalize_adj_matrix(e_rec, e_rec.shape[0], self.device)
            #print(new_adj[:10,:10])
            #norm_out = normalize_adj_matrix(e_out, e_out.shape[0], self.device)
            new_adj = self.config['lambda'] * norm_adj + (1 - self.config['lambda']) * e_rec

            #n_out = self.model.network(x, new_adj)
            #print(n_out[:10])

            smoothness, connectivity, sparsity = self.edge_criterion(new_adj, x)
            sm.append(smoothness.item())
            con.append(connectivity.item())
            spar.append(sparsity.item())
            e_loss = self.config['alpha'] * smoothness + self.config['beta'] * connectivity + self.config['gamma'] * sparsity
            #n_loss = self.network_criterion(input=n_out[idx_train], target=y_hot[idx_train])
            #print(' train loss edge: {}, network {}'.format(e_loss.item(), n_loss.item()))
            
            
            loss_edge.append(e_loss.item())
            #loss_pred.append(n_loss.item())
            #loss_total.append(e_loss.item() + n_loss.item())
            loss = e_loss

            self.model.optims.zero_grad()
            loss.backward()
            self.model.optims.step()

            #edge_module.eval()
            #network.eval()
            #val_accuracy = eval_classification(n_out[idx_val], n_out[idx_val])

        print('train end')
        """
        plt.plot(sm, 'g', label="smoothness")
        plt.plot(con, 'b', label="connectivity")
        plt.plot(spar, 'y', label="sparsity")
        plt.plot(loss_edge, 'r', label="total")
        plt.legend()
        plt.show()
        """