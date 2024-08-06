import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborLoader
import matplotlib.pyplot as plt
from ogb.nodeproppred import Evaluator
from sklearn.metrics import accuracy_score
import datetime as dt
import pandas as pd

from .utils.load_utils import load_network
from .utils.data_utils import topk_accuracy
from .utils.utils import initialize_logger
from .utils.noise import flip_label
from .models.model import NGNN
from .utils.losses import *

class PipelineSG(object):
    """
    Processing pipeline
    """
    def __init__(self, config):
        # Set metrics:
        self.device = config['device']

        # Data prep
        self.data, dataset = load_network(config)
        print('noise type and rate: {} {}'.format(config['noise_type'], config['noise_rate']))
        self.data.yhn, self.noise_mat = flip_label(self.data.y, dataset.num_classes, config['noise_type'], config['noise_rate'])
        self.noise_or_not = (self.data.y.squeeze() == self.data.yhn) #.int() # true if same lbl
        self.initial_edge_index = self.data.edge_index

        config['nbr_features'] = dataset.num_features #self.dataset.x.shape[-1]
        config['nbr_classes'] = dataset.num_classes #dataset.y.max().item() + 1
        config['nbr_nodes'] = dataset.x.shape[0]
        
        # Config
        self.config = config

        # Initialize the model
        self.criterion = CTLoss(self.device)
        
        train_idx = self.data.train_mask.nonzero().squeeze()
        val_idx = self.data.val_mask.nonzero().squeeze()
        test_idx = self.data.test_mask.nonzero().squeeze()
        self.split_idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        if self.config['batch_size_full']:
            self.config['batch_size'] = self.split_idx['train'].shape[0]
        print('train: {}, valid: {}, test: {}'.format(self.split_idx['train'].shape[0],self.split_idx['valid'].shape[0],self.split_idx['test'].shape[0]))
        print('batch size: {}'.format(self.config['batch_size']))
        
        # Logger and data loader
        date = dt.datetime.date(dt.datetime.now())
        self.output_name = 'dt{}{}_{}_id{}_{}_{}_{}_undirect_{}_noise_{}{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_tk{}_cttau{}_neigh{}{}___grid'.format(date.month,date.day,self.config['dataset_name'],self.config['batch_id'],self.config['train_type'],self.config['algo_type'],self.config['module'],self.config['undirected'],self.config['noise_type'],self.config['noise_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['max_epochs'],self.config['batch_size'],self.config['dropout'],self.config['ct_tk'],self.config['ct_tau'],self.config['nbr_neighbors'][0],self.config['nbr_neighbors'][1])#,self.config['nbr_neighbors'][2])
        self.logger = initialize_logger(self.config, self.output_name)
        

    def train_ct(self, train_loader, epoch, model1, optimizer1, model2, optimizer2):
        if not((epoch+1)%50) or ((epoch+1)==1):
            print('   Train epoch {}/{}'.format(epoch+1, self.config['max_epochs']))
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
            
            loss_1, loss_2, pure_ratio_1, pure_ratio_2, _, _, _, _  = self.criterion(out1, out2, yhn, self.rate_schedule[epoch], batch.n_id, self.noise_or_not)
            
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
        train_acc_1 = total_correct_1 / self.split_idx['train'].shape[0]
        train_acc_2 = total_correct_2 / self.split_idx['train'].shape[0]
        pure_ratio_1_list = total_ratio_1 / len(train_loader)
        pure_ratio_2_list = total_ratio_2 / len(train_loader)
        
        return train_loss_1, train_loss_2, train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list

    def test_planet(self, subgraph_loader, model):
        model.eval()

        with torch.no_grad():
            out = model.inference(self.data.x, subgraph_loader, self.device)
            
            y_true = self.data.y.cpu()
            y_pred = out.argmax(dim=-1, keepdim=True)

            train_acc = accuracy_score(y_true[self.split_idx['train']], y_pred[self.split_idx['train']])
            val_acc = accuracy_score(y_true[self.split_idx['valid']], y_pred[self.split_idx['valid']])
            test_acc = accuracy_score(y_true[self.split_idx['test']], y_pred[self.split_idx['test']])

        return train_acc, val_acc, test_acc

    def loop(self):
        print('loop')
        self.logger.info('{} RUNS grid search'.format(self.config['num_runs']))
        results_df = pd.DataFrame(columns=['undirect', 'nb', 'hid', 'tk', 'tau', 'mean', 'std'])

        for undirect in [True, False]:
            if undirect:
                self.data.edge_index = to_undirected(self.initial_edge_index)
            else:
                self.data.edge_index = self.initial_edge_index
            for nb in [[7,3], [10,5], [15,10]]:
                self.train_loader = NeighborLoader(
                    self.data,
                    input_nodes=self.split_idx['train'],
                    num_neighbors=nb,
                    batch_size=self.config['batch_size'],
                    shuffle=True,
                    num_workers=1,
                    persistent_workers=True
                )
                self.subgraph_loader = NeighborLoader(
                    self.data,
                    input_nodes=None,
                    num_neighbors=nb,
                    batch_size=4096,
                    num_workers=4,
                    persistent_workers=True,
                )
                #print('length train_loader: {}, subgraph_loader: {}'.format(len(self.train_loader),len(self.subgraph_loader)))
                for hid in [256, 512, 1024]:
                    self.model1 = NGNN(self.config['nbr_features'],hid,self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'])
                    self.model2 = NGNN(self.config['nbr_features'],hid,self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'])
                    for tk in [10,15,20]:
                        for tau in [0.1,0.2,0.4]:
                            self.rate_schedule = np.ones(self.config['max_epochs'])*self.config['noise_rate']*tau
                            self.rate_schedule[:tk] = np.linspace(0, self.config['noise_rate']*tau, tk)
                            
                            best_acc_ct = []
                            for i in range(self.config['num_runs']):
                                #self.logger.info('   Train nalgo')
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
                                test_acc_1_hist = []
                                test_acc_2_hist = []

                                for epoch in range(self.config['max_epochs']):
                                    train_loss_1, train_loss_2, _, _, pure_ratio_1_list, pure_ratio_2_list = self.train_ct(self.train_loader, epoch, self.model1.network.to(self.device), self.model1.optimizer, self.model2.network.to(self.device), self.model2.optimizer)
                                    train_acc_1, val_acc_1, test_acc_1 = self.test_planet(self.subgraph_loader, self.model1.network.to(self.device))
                                    train_acc_2, val_acc_2, test_acc_2 = self.test_planet(self.subgraph_loader, self.model2.network.to(self.device))

                                    train_loss_1_hist.append(train_loss_1), train_loss_2_hist.append(train_loss_2)
                                    train_acc_1_hist.append(train_acc_1), train_acc_2_hist.append(train_acc_2)
                                    pure_ratio_1_hist.append(pure_ratio_1_list), pure_ratio_2_hist.append(pure_ratio_2_list)
                                    val_acc_1_hist.append(val_acc_1), val_acc_2_hist.append(val_acc_2)
                                    test_acc_1_hist.append(test_acc_1), test_acc_2_hist.append(test_acc_2)
                                    
                                #self.logger.info('   RUN {} - best nalgo test acc1: {:.3f}   acc2: {:.3f}'.format(i+1,max(test_acc_1_hist),max(test_acc_2_hist)))
                                best_acc_ct.append(max(max(test_acc_1_hist),max(test_acc_2_hist)))
                            std, mean = torch.std_mean(torch.as_tensor(best_acc_ct))
                            self.logger.info('   undirect {}, nb {}, hid {}, tk {}, tau {} --- mean {:.3f} +- {:.3f} std'.format(undirect, nb, hid, tk, tau, mean,std))
                            
                            new_row = pd.DataFrame({'undirect': [undirect], 'nb': [nb], 'hid': [hid], 'tk': [tk], 'tau': [tau], 'mean': [mean], 'std': [std]})
                            results_df = pd.concat([results_df, new_row], ignore_index=True)
        top_results = results_df.sort_values(by='mean', ascending=False).head(7)
        self.logger.info(' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  RESULTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ')
        for i, row in top_results.iterrows():
            self.logger.info('mean {:.3f} +- {:.3f} std --- values undirect {}, nb {}, hid {}, tk {}, tau {}'.format(row['mean'], row['std'], row['undirect'], row['nb'], row['hid'], row['tk'], row['tau']))

        print('Done training')
        self.logger.info('Done training')
