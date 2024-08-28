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
        # Config
        self.config = config

        
        self.data, dataset = load_network(self.config)
        print('noise type and rate: {} {}'.format(self.config['noise_type'], self.config['noise_rate']))
        self.data.yhn, self.noise_mat = flip_label(self.data.y, dataset.num_classes, self.config['noise_type'], self.config['noise_rate'])
        self.noise_or_not = (self.data.y.squeeze() == self.data.yhn) #.int() # true if same lbl

        self.config['nbr_features'] = dataset.num_features #self.dataset.x.shape[-1]
        self.config['nbr_classes'] = dataset.num_classes #dataset.y.max().item() + 1
        self.config['nbr_nodes'] = dataset.x.shape[0]
        
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
        self.config['algo_type'] = 'coteaching'
        self.output_name = 'dt{}{}_{}_id{}_{}_{}_{}_noise_{}{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_tk{}_cttau{}_neigh{}{}___grid'.format(date.month,date.day,self.config['dataset_name'],self.config['batch_id'],self.config['train_type'],self.config['algo_type'],self.config['module'],self.config['noise_type'],self.config['noise_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['max_epochs'],self.config['batch_size'],self.config['dropout'],self.config['ct_tk'],self.config['ct_tau'],self.config['nbr_neighbors'][0],self.config['nbr_neighbors'][1])#,self.config['nbr_neighbors'][2])
        self.logger = initialize_logger(self.config, self.output_name)

        self.train_loader = NeighborLoader(
            self.data,
            input_nodes=self.split_idx['train'],
            num_neighbors=self.config['nbr_neighbors'],
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=1,
            persistent_workers=True
        )
        self.subgraph_loader = NeighborLoader(
            self.data,
            input_nodes=None,
            num_neighbors=self.config['nbr_neighbors'],
            batch_size=4096,
            num_workers=4,
            persistent_workers=True,
        )
        #print('length train_loader: {}, subgraph_loader: {}'.format(len(self.train_loader),len(self.subgraph_loader)))
        

    def train_ct(self, train_loader, epoch, model1, model2, optimizer):
        if not((epoch+1)%5) or ((epoch+1)==1):
            print('   Train epoch {}/{}'.format(epoch+1, self.config['max_epochs']))
        model1.train()
        model2.train()

        pure_ratio_1_list=[]
        pure_ratio_2_list=[]
        total_loss_1=0
        total_loss_2=0
        total_loss_cont_1=0
        total_loss_cont_2=0
        total_correct_1=0
        total_correct_2=0
        total_ratio_1=0
        total_ratio_2=0

        for batch in train_loader:
            batch = batch.to(self.device)

            # Only consider predictions and labels of seed nodes
            h_pure1, _, z_pure1, _, _, _ = model1(batch.x, batch.edge_index, noise_rate=self.config['spl_noise_rate_pos'], n_id=batch.n_id)
            h_pure2, _, z_pure2, _, _, _ = model2(batch.x, batch.edge_index, noise_rate=self.config['spl_noise_rate_pos'], n_id=batch.n_id)
            
            out1 = z_pure1[:batch.batch_size]
            out2 = z_pure2[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            yhn = batch.yhn[:batch.batch_size].squeeze()
            
            loss_1, loss_2, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update, ind_noisy_1, ind_noisy_2  = self.criterion(out1, out2, yhn, self.rate_schedule[epoch], batch.n_id, self.noise_or_not)
            
            if epoch > self.config['ct_tk']:
                # Rewire
                pos_edge, neg_edge = topk_rewire(h_pure1, batch.edge_index, self.device, k_percent=self.config['spl_rewire_rate'], directed=False)
                # Pos samples
                hedge_pure1, _, _, _, _, _ = model1(batch.x, pos_edge, noise_rate=self.config['spl_noise_rate_pos'], n_id=batch.n_id)
                hedge_pure2, _, _, _, _, _ = model2(batch.x, pos_edge, noise_rate=self.config['spl_noise_rate_pos'], n_id=batch.n_id)
                # Neg samples
                new_x = shuffle_pos(batch.x, device=self.device, prob=self.config['spl_noise_rate_neg'])
                _, _, _, hneg_noisy1, _, _ = model1(new_x, neg_edge, noise_rate=self.config['spl_noise_rate_neg'], n_id=batch.n_id)
                _, _, _, hneg_noisy2, _, _ = model2(new_x, neg_edge, noise_rate=self.config['spl_noise_rate_neg'], n_id=batch.n_id)
                # Contrastive
                logits_pa1, logits_n1 = self.discriminator(h_pure1[ind_noisy_1], hedge_pure1[ind_noisy_1], hneg_noisy1[ind_noisy_1])
                logits_pa2, logits_n2 = self.discriminator(h_pure2[ind_noisy_2], hedge_pure2[ind_noisy_2], hneg_noisy2[ind_noisy_2])
                #logits_pa1, logits_n1 = self.discriminator(h_pure1[:batch.batch_size], hedge_pure1[:batch.batch_size], hneg_noisy1[:batch.batch_size])
                #logits_pa2, logits_n2 = self.discriminator(h_pure2[:batch.batch_size], hedge_pure2[:batch.batch_size], hneg_noisy2[:batch.batch_size])
                loss_cont1 = self.cont_criterion(logits_pa1, logits_n1)
                loss_cont2 = self.cont_criterion(logits_pa2, logits_n2)

                loss = loss_1 + loss_2 + self.config['spl_cont_beta'] * loss_cont1 + self.config['spl_cont_beta'] * loss_cont2
            else:
                loss = loss_1 + loss_2
                loss_cont1 = 0
                loss_cont2 = 0
            total_loss_1 += float(loss_1)
            total_loss_2 += float(loss_2)
            total_loss_cont_1 += float(loss_cont1)
            total_loss_cont_2 += float(loss_cont2)
            total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())
            total_ratio_1 += (100*pure_ratio_1)
            total_ratio_2 += (100*pure_ratio_2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss_1 = total_loss_1 / len(train_loader)
        train_loss_2 = total_loss_2 / len(train_loader)
        train_loss_cont_1 = total_loss_cont_1 / len(train_loader)
        train_loss_cont_2 = total_loss_cont_2 / len(train_loader)
        train_acc_1 = total_correct_1 / self.split_idx['train'].size(0)
        train_acc_2 = total_correct_2 / self.split_idx['train'].size(0)
        pure_ratio_1_list = total_ratio_1 / len(train_loader)
        pure_ratio_2_list = total_ratio_2 / len(train_loader)
        
        return train_loss_1, train_loss_2, train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list, train_loss_cont_1, train_loss_cont_2


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

        
        for drop in [0.5]:
            for lay in [2]:
                for hid in [512]:
                    for tk in [25]:
                        for beta in [0.05,0.1,0.2]:
                            best_acc_ct = []
                            for i in range(self.config['num_runs']):
                                # Initialize the model
                                hid = self.config['hidden_size']
                                lay = self.config['num_layers']
                                drop = self.config['dropout']
                                tk = self.config['ct_tk']
                                tau = self.config['ct_tau']
                                self.config['spl_cont_beta'] = beta
                                self.model1 = NGNN(self.config['nbr_features'],hid,self.config['nbr_classes'],lay,drop,self.config['learning_rate'],self.config['optimizer'],self.config['module'])
                                self.model2 = NGNN(self.config['nbr_features'],hid,self.config['nbr_classes'],lay,drop,self.config['learning_rate'],self.config['optimizer'],self.config['module'])
                                self.criterion = CTLoss(self.device)
                                self.rate_schedule = np.ones(self.config['max_epochs'])*self.config['noise_rate']*tau
                                self.rate_schedule[:tk] = np.linspace(0, self.config['noise_rate']*tau, tk)
                                
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
                                    train_loss_1, train_loss_2, train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list, train_loss_cont_1, train_loss_cont_2 = self.train_ct(self.train_loader, epoch, self.model1.network.to(self.device), self.model1.optimizer, self.model2.network.to(self.device), self.model2.optimizer)
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
                            self.logger.info('   drop {}, lay {}, hid {}, tk {}, beta {} --- mean {:.3f} +- {:.3f} std'.format(drop, lay, hid, tk, beta, mean,std))
                            
                            new_row = pd.DataFrame({'drop': [drop], 'lay': [lay],  'hid': [hid], 'tk': [tk], 'beta': [beta], 'mean': [mean], 'std': [std]})
                            results_df = pd.concat([results_df, new_row], ignore_index=True)
        top_results = results_df.sort_values(by='mean', ascending=False).head(3)
        self.logger.info(' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  RESULTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ')
        for i, row in top_results.iterrows():
            self.logger.info('mean {:.3f} +- {:.3f} std --- values drop {}, lay {}, hid {}, tk {}, beta {}'.format(row['mean'], row['std'], row['drop'], row['lay'], row['hid'], row['tk'], row['beta']))

        print('Done training')
        self.logger.info('Done training')
