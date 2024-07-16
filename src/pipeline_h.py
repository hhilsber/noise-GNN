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
from .utils.augmentation import topk_rewire
from .utils.utils import initialize_logger
from .utils.noise import flip_label
from .models.model import NGNN
from .models.layers.gcn import GCN
from .utils.losses import *

class PipelineH(object):
    """
    Processing pipeline
    """
    def __init__(self, config):
        # Set metrics:
        self.device = config['device']

        # Data prep
        self.dataset = load_network(config)
        self.data = self.dataset[0]
        print('noise type and rate: {} {}'.format(config['noise_type'], config['noise_rate']))
        self.data.yhn, noise_mat = flip_label(self.data.y, self.dataset.num_classes, config['noise_type'], config['noise_rate'])
        self.noise_or_not = (self.data.y.squeeze() == self.data.yhn) #.int() # true if same lbl
        
        config['nbr_features'] = self.dataset.num_features #self.dataset.x.shape[-1]
        config['nbr_classes'] = self.dataset.num_classes #dataset.y.max().item() + 1
        config['nbr_nodes'] = self.dataset.x.shape[0]

        # Config
        self.config = config

        # Initialize the model
        self.model1 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'],self.config['weight_decay'])
        self.model2 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'],self.config['weight_decay'])
        self.pseudo_gcn = GCN(self.config['hidden_size'],self.config['nbr_classes'])
        #self.pseudo_optim = torch.optim.Adam(self.pseudo_gcn.parameters(),lr=config['learning_rate'])

        self.optimizer = torch.optim.Adam(list(self.model1.network.parameters()) + list(self.model2.network.parameters())+ list(self.pseudo_gcn.parameters()),lr=config['learning_rate'],weight_decay=config['weight_decay'])

        self.criterion = CTLoss(self.device)
        self.rate_schedule = np.ones(self.config['max_epochs'])*self.config['noise_rate']*self.config['ct_tau']
        self.rate_schedule[:self.config['ct_tk']] = np.linspace(0, self.config['noise_rate']**self.config['ct_exp'], self.config['ct_tk'])

        # Split data set
        self.split_idx = self.dataset.get_idx_split()

        print('train: {}, valid: {}, test: {}'.format(self.split_idx['train'].shape[0],self.split_idx['valid'].shape[0],self.split_idx['test'].shape[0]))

        # Logger and data loader
        date = dt.datetime.date(dt.datetime.now())
        self.output_name = 'dt{}{}_{}_id{}_{}_{}_{}_algo_{}_split_{}_noise_{}{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_tk{}_cttau{}_neigh{}{}{}'.format(date.month,date.day,self.config['dataset_name'],self.config['batch_id'],self.config['train_type'],self.config['algo_type'],self.config['module'],self.config['compare_loss'],self.config['original_split'],self.config['noise_type'],self.config['noise_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['max_epochs'],self.config['batch_size'],self.config['dropout'],self.config['ct_tk'],self.config['ct_tau'],self.config['nbr_neighbors'][0],self.config['nbr_neighbors'][1],self.config['nbr_neighbors'][2])
        self.logger = initialize_logger(self.config, self.output_name)
        #np.save('../out_nmat/' + self.output_name + '.npy', noise_mat)

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
        
        self.test_loader = NeighborLoader(
            self.data,
            input_nodes=self.split_idx['test'],
            num_neighbors=self.config['nbr_neighbors'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            persistent_workers=True
        )

        #########################################################################################


    def train_ct(self, train_loader, epoch, model1, model2, pseudo_gcn, optimizer):
        if not((epoch+1)%5) or ((epoch+1)==1):
            print('   Train epoch {}/{}'.format(epoch+1, self.config['max_epochs']))
        model1.train()
        model2.train()

        pure_ratio_1_list=[]
        pure_ratio_2_list=[]
        total_loss_1=0
        total_loss_2=0
        total_correct_1=0
        total_correct_2=0
        total_correct_pl1=0
        total_correct_pl2=0
        total_ratio_1=0
        total_ratio_2=0

        total_loss_pred=0
        total_loss_add=0

        for batch in train_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out1, h1 = model1(batch.x, batch.edge_index)
            out2, h2 = model2(batch.x, batch.edge_index)
            out1 = out1[:batch.batch_size]
            out2 = out2[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            yhn = batch.yhn[:batch.batch_size].squeeze()

            loss_ct_1, loss_ct_2, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update, ind_noisy_1, ind_noisy_2  = self.criterion(out1, out2, yhn, self.rate_schedule[epoch], batch.n_id, self.noise_or_not)
            
            if (epoch > 0):
                new_edge1 = topk_rewire(batch.x, batch.edge_index, self.device, k_percent=0.2)
                new_edge2 = new_edge1 #topk_rewire(batch.x, batch.edge_index, self.device, k_percent=0.2)

                pseudo_lbl_1 = pseudo_gcn(h1, new_edge1)[:batch.batch_size]
                pred_1 = F.softmax(pseudo_lbl_1,dim=1).detach()
                pseudo_lbl_2 = pseudo_gcn(h2, new_edge2)[:batch.batch_size]
                pred_2 = F.softmax(pseudo_lbl_2,dim=1).detach()
                
                pred_model_1 = F.softmax(out1,dim=1)
                pred_model_2 = F.softmax(out2,dim=1)
                # loss from pseudo labels
                loss_add = (-torch.sum(pred_1[ind_noisy_1] * torch.log(pred_model_1[ind_noisy_1]), dim=1)).mean() \
                             + (-torch.sum(pred_2[ind_noisy_2] * torch.log(pred_model_2[ind_noisy_2]), dim=1)).mean()

                loss_pred = F.cross_entropy(pseudo_lbl_1[ind_1_update], yhn[ind_1_update]) \
                             + F.cross_entropy(pseudo_lbl_2[ind_2_update], yhn[ind_2_update])

                #loss_pseudo_1 = F.cross_entropy(out1[ind_noisy_1], pseudo_lbl_1[ind_noisy_1])
                #loss_pseudo_2 = F.cross_entropy(out2[ind_noisy_2], pseudo_lbl_2[ind_noisy_2])
                beta = 0.8
                loss = loss_ct_1 + loss_ct_2 + loss_pred + beta * loss_add
            else:
                loss_1 = loss_ct_1
                loss_2 = loss_ct_2
                loss = loss_ct_1 + loss_ct_2
                loss_pred = 0
                loss_add = 0
                pseudo_lbl_1 = out1
                pseudo_lbl_2 = out2
           

            total_loss_1 += float(loss_ct_1)
            total_loss_2 += float(loss_ct_2)
            total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())
            total_correct_pl1 += int(pseudo_lbl_1.argmax(dim=-1).eq(y).sum())
            total_correct_pl2 += int(pseudo_lbl_2.argmax(dim=-1).eq(y).sum())
            total_ratio_1 += (100*pure_ratio_1)
            total_ratio_2 += (100*pure_ratio_2)

            total_loss_pred += float(loss_pred)
            total_loss_add += float(loss_add)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        train_loss_1 = total_loss_1 / len(train_loader)
        train_loss_2 = total_loss_2 / len(train_loader)
        train_acc_1 = total_correct_1 / self.split_idx['train'].size(0)
        train_acc_2 = total_correct_2 / self.split_idx['train'].size(0)
        train_acc_pl1 = total_correct_pl1 / self.split_idx['train'].size(0)
        train_acc_pl2 = total_correct_pl2 / self.split_idx['train'].size(0)
        pure_ratio_1_list = total_ratio_1 / len(train_loader)
        pure_ratio_2_list = total_ratio_2 / len(train_loader)
        
        train_loss_pred = total_loss_pred / len(train_loader)
        train_loss_add = total_loss_add / len(train_loader)
        return train_loss_1, train_loss_2, train_acc_1, train_acc_2, train_acc_pl1, train_acc_pl2, pure_ratio_1_list, pure_ratio_2_list, train_loss_pred, train_loss_add

    def evaluate_ct(self, valid_loader, model1, model2):
        model1.eval()
        model2.eval()

        total_correct_1 = 0
        total_correct_2 = 0
        
        for batch in valid_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out1, _ = model1(batch.x, batch.edge_index, training=False)
            out2, _ = model2(batch.x, batch.edge_index, training=False)
            out1 = out1[:batch.batch_size]
            out2 = out2[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()

            total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())
        val_acc_1 = total_correct_1 / self.split_idx['valid'].size(0)
        val_acc_2 = total_correct_2 / self.split_idx['valid'].size(0)
        return val_acc_1, val_acc_2
    
    def test_ct(self, test_loader, model1, model2):
        model1.eval()
        model2.eval()

        total_correct_1 = 0
        total_correct_2 = 0

        for batch in test_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out1, _ = model1(batch.x, batch.edge_index, training=False)
            out2, _ = model2(batch.x, batch.edge_index, training=False)
            out1 = out1[:batch.batch_size]
            out2 = out2[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()

            total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())
        test_acc_1 = total_correct_1 / self.split_idx['test'].size(0) #self.test_size
        test_acc_2 = total_correct_2 / self.split_idx['test'].size(0)
        return test_acc_1, test_acc_2


    def loop(self):
        print('loop')
        
        print('Train nalgo')
        self.logger.info('Train nalgo')
        #self.model1.network.reset_parameters()
        #self.model2.network.reset_parameters()

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
        train_acc_pl1_hist = []
        train_acc_pl2_hist = []

        train_loss_pred_hist = []
        train_loss_add_hist = []

        for epoch in range(self.config['max_epochs']):
            train_loss_1, train_loss_2, train_acc_1, train_acc_2, train_acc_pl1, train_acc_pl2, pure_ratio_1_list, pure_ratio_2_list, train_loss_pred, train_loss_add = self.train_ct(self.train_loader, epoch, self.model1.network.to(self.device), self.model2.network.to(self.device), self.pseudo_gcn.to(self.device), self.optimizer)

            train_loss_1_hist.append(train_loss_1), train_loss_2_hist.append(train_loss_2)
            train_acc_1_hist.append(train_acc_1), train_acc_2_hist.append(train_acc_2)
            train_acc_pl1_hist.append(train_acc_pl1), train_acc_pl2_hist.append(train_acc_pl2)
            pure_ratio_1_hist.append(pure_ratio_1_list), pure_ratio_2_hist.append(pure_ratio_2_list)
            
            train_loss_pred_hist.append(train_loss_pred), train_loss_add_hist.append(train_loss_add)

            val_acc_1, val_acc_2 = self.evaluate_ct(self.valid_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
            val_acc_1_hist.append(val_acc_1), val_acc_2_hist.append(val_acc_2)
            
            test_acc_1, test_acc_2 = self.test_ct(self.test_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
            test_acc_1_hist.append(test_acc_1), test_acc_2_hist.append(test_acc_2)
            self.logger.info('   Train epoch {}/{} --- acc t1: {:.3f} t2: {:.3f} v1: {:.3f} v2: {:.3f} tst1: {:.3f} tst2: {:.3f}'.format(epoch+1,self.config['max_epochs'],train_acc_1,train_acc_2,val_acc_1,val_acc_2,test_acc_1,test_acc_2))
        
            
        print('Done training')
        self.logger.info('Done training')

        self.logger.info('Best test acc1: {:.3f}   acc2: {:.3f}'.format(max(test_acc_1_hist),max(test_acc_2_hist)))
        print('Done')

        if self.config['do_plot']:
            fig, axs = plt.subplots(4, 1, figsize=(10, 15))
            
            axs[0].axhline(y=0.55, color='grey', linestyle='--')
            line1, = axs[0].plot(train_acc_1_hist, 'blue', label="train_acc_1_hist")
            line2, = axs[0].plot(train_acc_2_hist, 'darkgreen', label="train_acc_2_hist")
            line3, = axs[0].plot(val_acc_1_hist, 'purple', label="val_acc_1_hist")
            line4, = axs[0].plot(val_acc_2_hist, 'darkseagreen', label="val_acc_2_hist")
            line5, = axs[0].plot(test_acc_1_hist, 'deepskyblue', label="test_acc_1_hist")
            line6, = axs[0].plot(test_acc_2_hist, 'chartreuse', label="test_acc_2_hist")

            axs[1].plot(train_acc_pl1_hist, 'blue', label="train_acc_pl1_hist")
            axs[1].plot(train_acc_pl2_hist, 'darkgreen', label="train_acc_pl2_hist")

            axs[2].plot(pure_ratio_1_hist, 'blue', label="pure_ratio_1_hist")
            axs[2].plot(pure_ratio_2_hist, 'darkgreen', label="pure_ratio_2_hist")
            axs[2].legend()

            axs[3].plot(train_loss_1_hist, 'blue', label="train_loss_1_hist")
            axs[3].plot(train_loss_2_hist, 'darkgreen', label="train_loss_2_hist")
            axs[3].plot(train_loss_pred_hist, 'red', label="train_loss_pred_hist")
            axs[3].plot(train_loss_add_hist, 'cyan', label="train_loss_add_hist")
            
            axs[0].legend(handles=[line1, line2, line3, line4, line5, line6], loc='upper left', bbox_to_anchor=(1.05, 1))
            
            axs[0].set_title('Plot 1')
            axs[1].set_title('Plot 2')
            axs[2].set_title('Plot 3')
            axs[3].legend()
            axs[3].set_title('Plot 4')

            plt.tight_layout()
            #plt.show()
            plot_name = '../out_plots/coteaching_expe/' + self.output_name + '.png'
            plt.savefig(plot_name)

