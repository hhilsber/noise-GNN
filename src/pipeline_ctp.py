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
from .utils.losses import *

class PipelineCTP(object):
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
        if self.config['train_type'] in ['nalgo','both']:
            self.model1 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'],nbr_nodes=self.config['nbr_nodes'])
            self.model2 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'],nbr_nodes=self.config['nbr_nodes'])
            self.criterion = CTLoss(self.device)
            self.rate_schedule = np.ones(self.config['max_epochs'])*self.config['noise_rate']*self.config['ct_tau']
            self.rate_schedule[:self.config['ct_tk']] = np.linspace(0, self.config['noise_rate']**self.config['ct_exp'], self.config['ct_tk'])
        if self.config['train_type'] in ['baseline','both']:
            self.model_c = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],"sage")
        
        self.split_idx = self.dataset.get_idx_split()
        print('train: {}, valid: {}, test: {}'.format(self.split_idx['train'].shape[0],self.split_idx['valid'].shape[0],self.split_idx['test'].shape[0]))

        # Logger and data loader
        date = dt.datetime.date(dt.datetime.now())
        self.output_name = 'dt{}{}_{}_id{}_{}_{}_{}_noise_{}{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_tk{}_cttau{}_neigh{}{}'.format(date.month,date.day,self.config['dataset_name'],self.config['batch_id'],self.config['algo_type'],self.config['train_type'],self.config['module'],self.config['noise_type'],self.config['noise_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['max_epochs'],self.config['batch_size'],self.config['dropout'],self.config['ct_tk'],self.config['ct_tau'],self.config['nbr_neighbors'][0],self.config['nbr_neighbors'][1])
        self.logger = initialize_logger(self.config, self.output_name)

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

    def train_ct(self, train_loader, epoch, model1, optimizer1, model2, optimizer2):
        if not((epoch+1)%5) or ((epoch+1)==1):
            print('   Train epoch {}/{}'.format(epoch+1, self.config['max_epochs']))
        model1.train()
        model2.train()

        pure_ratio_1_list = []
        pure_ratio_2_list = []
        total_loss_1 = 0
        total_loss_2 = 0
        total_correct_1 = 0
        total_correct_2 = 0
        total_ratio_1 = 0
        total_ratio_2 = 0

        total_loss_nal_1 = 0
        total_loss_nal_2 = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            # Rewire 
            new_edge = topk_rewire(batch.x, batch.edge_index, self.device, k_percent=0.1)

            # Only consider predictions and labels of seed nodes
            x_pure1, y_pure1, z_pure1, x_noisy1, y_noisy1, z_noisy1 = model1(batch.x, batch.edge_index, new_edge, noisy_rate=self.config['spl_noise'], n_id=batch.n_id)
            x_pure2, y_pure2, z_pure2, x_noisy2, y_noisy2, z_noisy2 = model2(batch.x, batch.edge_index, new_edge, noisy_rate=self.config['spl_noise'], n_id=batch.n_id)
            out1 = z_pure1[:batch.batch_size]
            out2 = z_pure2[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            yhn = batch.yhn[:batch.batch_size].squeeze()
            
            loss_ct_1, loss_ct_2, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update, ind_noisy_1, ind_noisy_2 = self.criterion(out1, out2, yhn, self.rate_schedule[epoch], batch.n_id, self.noise_or_not)
            
            if (epoch > 0):
                """
                loss_nal_1 = neighbor_align_batch(batch.edge_index, batch.x, out1_full, ind_noisy_1, batch_size=batch.batch_size, device=self.device)
                loss_nal_2 = neighbor_align_batch(batch.edge_index, batch.x, out2_full, ind_noisy_2, batch_size=batch.batch_size, device=self.device)
                ###
                beta = 0.5
                loss_1 = loss_ct_1 + beta * loss_nal_1
                loss_2 = loss_ct_2 + beta * loss_nal_2"""
                
                with torch.no_grad():
                    w1 = get_uncertainty_batch(batch.edge_index, y_pure1, nbr_classes=self.config['nbr_classes'], device=self.device)
                    w2 = get_uncertainty_batch(batch.edge_index, y_pure2, nbr_classes=self.config['nbr_classes'], device=self.device)
                
                loss_cr1 = fix_cr(y_pure1, y_noisy1, ind_noisy_1, batch_size=batch.batch_size, name='ce', w=w1)
                loss_cr2 = fix_cr(y_pure2, y_noisy2, ind_noisy_2, batch_size=batch.batch_size, name='ce', w=w2)
                beta = 1.0
                loss_1 = loss_ct_1 + beta * loss_cr1
                loss_2 = loss_ct_2 + beta * loss_cr2
            else:
                loss_1 = loss_ct_1
                loss_2 = loss_ct_2
                loss_cr1 = 0
                loss_cr2 = 0
            
            

            total_loss_1 += float(loss_1)
            total_loss_2 += float(loss_2)
            total_loss_nal_1 += float(loss_cr1)
            total_loss_nal_2 += float(loss_cr2)

            total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())
            total_ratio_1 += (100*pure_ratio_1)
            total_ratio_2 += (100*pure_ratio_2)
            
            optimizer1.zero_grad()
            loss_1.backward()
            #loss_1.backward(retain_graph=True)
            optimizer1.step()

            
            
            optimizer2.zero_grad()
            loss_2.backward()
            #loss_2.backward(retain_graph=True)
            optimizer2.step()

        train_loss_1 = total_loss_1 / len(train_loader)
        train_loss_2 = total_loss_2 / len(train_loader)
        train_loss_nal_1 = total_loss_nal_1 / len(train_loader)
        train_loss_nal_2 = total_loss_nal_2 / len(train_loader)

        train_acc_1 = total_correct_1 / self.split_idx['train'].size(0)
        train_acc_2 = total_correct_2 / self.split_idx['train'].size(0)
        pure_ratio_1_list = total_ratio_1 / len(train_loader)
        pure_ratio_2_list = total_ratio_2 / len(train_loader)
        
        return train_loss_1, train_loss_2, train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list, train_loss_nal_1, train_loss_nal_2
    

    def evaluate_ct(self, valid_loader, model1, model2):
        model1.eval()
        model2.eval()

        total_correct_1 = 0
        total_correct_2 = 0
        
        for batch in valid_loader:
            batch = batch.to(self.device)
            #
            new_edge = topk_rewire(batch.x, batch.edge_index, self.device, k_percent=0.2)

            # Only consider predictions and labels of seed nodes
            #out1 = model1(batch.x, batch.edge_index)[:batch.batch_size]
            #out2 = model2(batch.x, batch.edge_index)[:batch.batch_size]
            x_pure1, y_pure1, z_pure1, x_noisy1, y_noisy1, z_noisy1 = model1(batch.x, batch.edge_index, new_edge, n_id=batch.n_id)
            x_pure2, y_pure2, z_pure2, x_noisy2, y_noisy2, z_noisy2 = model2(batch.x, batch.edge_index, new_edge, n_id=batch.n_id)
            out1 = z_pure1[:batch.batch_size]
            out2 = z_pure2[:batch.batch_size]
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
            #
            #new_edge = topk_rewire(batch.x, batch.edge_index, self.device, k_percent=0.2)

            # Only consider predictions and labels of seed nodes
            x_pure1, y_pure1, z_pure1, x_noisy1, y_noisy1, z_noisy1 = model1(batch.x, batch.edge_index, batch.edge_index, n_id=batch.n_id)
            x_pure2, y_pure2, z_pure2, x_noisy2, y_noisy2, z_noisy2 = model2(batch.x, batch.edge_index, batch.edge_index, n_id=batch.n_id)
            out1 = z_pure1[:batch.batch_size]
            out2 = z_pure2[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()

            total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())
        test_acc_1 = total_correct_1 / self.split_idx['test'].size(0) #self.test_size
        test_acc_2 = total_correct_2 / self.split_idx['test'].size(0)
        return test_acc_1, test_acc_2

    def train(self, train_loader, epoch, model, optimizer):
        if not((epoch+1)%5) or ((epoch+1)==1):
            print('   Train epoch {}/{}'.format(epoch+1, self.config['max_epochs']))
            #print('     loss = F.cross_entropy(out, y)')
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
        #train_acc = total_correct / self.train_wo_noise.size(0)
        return train_loss, train_acc

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
    
    def test(self, test_loader, model):
        model.eval()

        total_correct = 0
        for batch in test_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()

            total_correct += int(out.argmax(dim=-1).eq(y).sum())
        test_acc = total_correct / self.split_idx['test'].size(0) #self.test_size #
        return test_acc
    
    def loop(self):
        print('loop')
        
        if self.config['train_type'] in ['nalgo','both']:
            print('Train nalgo')
            self.logger.info('Train nalgo')

            train_loss_1_hist = []
            train_loss_2_hist = []
            nal_loss_1_hist = []
            nal_loss_2_hist = []
            train_acc_1_hist = []
            train_acc_2_hist = []
            pure_ratio_1_hist = []
            pure_ratio_2_hist = []
            val_acc_1_hist = []
            val_acc_2_hist = []
            test_acc_1_hist = []
            test_acc_2_hist = []

            for epoch in range(self.config['max_epochs']):
                train_loss_1, train_loss_2, train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list, train_loss_nal_1, train_loss_nal_2 = self.train_ct(self.train_loader, epoch, self.model1.network.to(self.device), self.model1.optimizer, self.model2.network.to(self.device), self.model2.optimizer)
                print('{:.3f} --- {:.3f}'.format(train_loss_nal_1, train_loss_nal_2))
                train_loss_1_hist.append(train_loss_1), train_loss_2_hist.append(train_loss_2)
                nal_loss_1_hist.append(train_loss_nal_1), nal_loss_2_hist.append(train_loss_nal_2)
                train_acc_1_hist.append(train_acc_1), train_acc_2_hist.append(train_acc_2)
                pure_ratio_1_hist.append(pure_ratio_1_list), pure_ratio_2_hist.append(pure_ratio_2_list)
                
                val_acc_1, val_acc_2 = self.evaluate_ct(self.valid_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
                val_acc_1_hist.append(val_acc_1), val_acc_2_hist.append(val_acc_2)
                
                test_acc_1, test_acc_2 = self.test_ct(self.test_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
                test_acc_1_hist.append(test_acc_1), test_acc_2_hist.append(test_acc_2)
                self.logger.info('   Train epoch {}/{} --- acc t1: {:.3f} t2: {:.3f} v1: {:.3f} v2: {:.3f} tst1: {:.3f} tst2: {:.3f}'.format(epoch+1,self.config['max_epochs'],train_acc_1,train_acc_2,val_acc_1,val_acc_2,test_acc_1,test_acc_2))
        
        if self.config['train_type'] in ['baseline','both']:
                print('Train baseline')
                self.logger.info('Train baseline')
                #self.model_c.network.reset_parameters()

                train_loss_hist = []
                train_acc_hist = []
                val_acc_hist = []
                test_acc_hist = []
                for epoch in range(self.config['max_epochs']):
                    train_loss, train_acc = self.train(self.train_loader, epoch, self.model_c.network.to(self.device), self.model_c.optimizer)
                    train_loss_hist.append(train_loss)
                    train_acc_hist.append(train_acc)

                    val_acc = self.evaluate(self.valid_loader, self.model_c.network.to(self.device))
                    val_acc_hist.append(val_acc)
                    test_acc = self.test(self.test_loader, self.model_c.network.to(self.device))
                    test_acc_hist.append(test_acc)
                    self.logger.info('   Train epoch {}/{} --- acc t: {:.4f} v: {:.4f} tst: {:.4f}'.format(epoch+1,self.config['max_epochs'],train_acc,val_acc,test_acc))
        
        if self.config['train_type'] in ['nalgo','both']:
            self.logger.info('Best test acc1: {:.3f}   acc2: {:.3f}'.format(max(test_acc_1_hist),max(test_acc_2_hist)))
        if self.config['train_type'] in ['baseline','both']:
            self.logger.info('Best baseline test acc: {:.3f}'.format(max(test_acc_hist)))
        print('Done')

        if self.config['do_plot']:
            fig, axs = plt.subplots(3, 1, figsize=(10, 15))
            
            #axs[0].axhline(y=0.8, color='grey', linestyle='--')
            axs[0].axhline(y=0.55, color='grey', linestyle='--')
            if self.config['train_type'] in ['nalgo','both']:
                line1, = axs[0].plot(train_acc_1_hist, 'blue', label="train_acc_1_hist")
                line2, = axs[0].plot(train_acc_2_hist, 'darkgreen', label="train_acc_2_hist")
                line3, = axs[0].plot(val_acc_1_hist, 'purple', label="val_acc_1_hist")
                line4, = axs[0].plot(val_acc_2_hist, 'darkseagreen', label="val_acc_2_hist")
                line5, = axs[0].plot(test_acc_1_hist, 'deepskyblue', label="test_acc_1_hist")
                line6, = axs[0].plot(test_acc_2_hist, 'chartreuse', label="test_acc_2_hist")
                axs[1].plot(pure_ratio_1_hist, 'blue', label="pure_ratio_1_hist")
                axs[1].plot(pure_ratio_2_hist, 'darkgreen', label="pure_ratio_2_hist")
                axs[1].legend()

                axs[2].plot(train_loss_1_hist, 'blue', label="train_loss_1_hist")
                axs[2].plot(train_loss_2_hist, 'darkgreen', label="train_loss_2_hist")
                axs[2].plot(nal_loss_1_hist, 'aqua', label="nal_loss_1_hist")
                axs[2].plot(nal_loss_2_hist, 'lawngreen', label="nal_loss_2_hist")
                
            if self.config['train_type'] in ['baseline','both']:
                line7, = axs[0].plot(train_acc_hist, 'red', label="train_acc_hist")
                line8, = axs[0].plot(val_acc_hist, 'tomato', label="val_acc_hist")
                line9, = axs[0].plot(test_acc_hist, 'deeppink', label="test_acc_hist")

                axs[2].plot(train_loss_hist, 'red', label="train_loss_hist")
            
            if self.config['train_type'] in ['nalgo']:
                axs[0].legend(handles=[line1, line2, line3, line4, line5, line6], loc='upper left', bbox_to_anchor=(1.05, 1))
            elif self.config['train_type'] in ['baseline']:
                axs[0].legend(handles=[line7, line8, line9], loc='upper left', bbox_to_anchor=(1.05, 1))
            else:
                axs[0].legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9], loc='upper left', bbox_to_anchor=(1.05, 1))
            
            axs[0].set_title('Plot 1')
            axs[1].set_title('Plot 2')
            axs[2].legend()
            axs[2].set_title('Plot 3')

            plt.tight_layout()
            #plt.show()
            plot_name = '../out_plots/ctp2/' + self.output_name + '.png'
            plt.savefig(plot_name)

