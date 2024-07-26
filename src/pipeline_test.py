import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, NeighborSampler
import matplotlib.pyplot as plt
from ogb.nodeproppred import Evaluator
import datetime as dt

from .utils.load_utils import load_network
from .utils.data_utils import Jensen_Shannon, Discriminator_innerprod, BCEExeprtLoss
from .utils.augmentation import *
from .utils.utils import initialize_logger
from .utils.noise import flip_label
from .models.model import NGNN
from .utils.losses import *

class PipelineTE(object):
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
        self.data.yhn, self.noise_mat = flip_label(self.data.y, self.dataset.num_classes, config['noise_type'], config['noise_rate'])
        self.noise_or_not = (self.data.y.squeeze() == self.data.yhn) #.int() # true if same lbl
        self.split_idx = self.dataset.get_idx_split()
        print('train: {}, valid: {}, test: {}'.format(self.split_idx['train'].shape[0],self.split_idx['valid'].shape[0],self.split_idx['test'].shape[0]))

        config['nbr_features'] = self.dataset.num_features #self.dataset.x.shape[-1]
        config['nbr_classes'] = self.dataset.num_classes #dataset.y.max().item() + 1
        config['nbr_nodes'] = self.dataset.x.shape[0]

        # Config
        self.config = config

        # Initialize the model
        if self.config['train_type'] in ['nalgo','both']:
            #self.model1 = NGNN(config)
            #self.model2 = NGNN(config)
            self.model1 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'],nbr_nodes=self.config['nbr_nodes'])
            self.model2 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'],nbr_nodes=self.config['nbr_nodes'])
            if self.config['algo_type'] == 'coteaching':
                self.criterion = CTLoss(self.device)
            elif self.config['algo_type'] == 'codi':
                self.criterion = CoDiLoss(self.device, self.config['co_lambda'])
            self.rate_schedule = np.ones(self.config['max_epochs'])*self.config['noise_rate']*self.config['ct_tau']
            self.rate_schedule[:self.config['ct_tk']] = np.linspace(0, self.config['noise_rate']**self.config['ct_exp'], self.config['ct_tk'])
        self.optimizer = torch.optim.Adam(list(self.model1.network.parameters()) + list(self.model2.network.parameters()),lr=config['learning_rate'])

        if self.config['train_type'] in ['baseline','both']:
            self.model_c = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],"sage")
        
        # Contrastive
        self.discriminator = Discriminator_innerprod()
        self.cont_criterion = BCEExeprtLoss(self.config['batch_size'])
        
        # Logger and data loader
        date = dt.datetime.date(dt.datetime.now())
        self.output_name = 'dt{}{}_{}_id{}_{}_{}_{}_algo_{}_split_{}_noise_{}{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_tk{}_cttau{}_neigh{}{}'.format(date.month,date.day,self.config['dataset_name'],self.config['batch_id'],self.config['train_type'],self.config['algo_type'],self.config['module'],self.config['compare_loss'],self.config['original_split'],self.config['noise_type'],self.config['noise_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['max_epochs'],self.config['batch_size'],self.config['dropout'],self.config['ct_tk'],self.config['ct_tau'],self.config['nbr_neighbors'][0],self.config['nbr_neighbors'][1])#,self.config['nbr_neighbors'][2])
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
        print(len(self.test_loader))


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
            h_pure1, _, z_pure1, h_noisy1, _, z_noisy1 = model1(batch.x, batch.edge_index, noise_rate=self.config['spl_noise_rate_pos'], n_id=batch.n_id)
            h_pure2, _, z_pure2, h_noisy2, _, z_noisy2 = model2(batch.x, batch.edge_index, noise_rate=self.config['spl_noise_rate_pos'], n_id=batch.n_id)
            
            out1 = z_pure1[:batch.batch_size]
            out2 = z_pure2[:batch.batch_size]
            noisy1 = z_noisy1[:batch.batch_size]
            noisy2 = z_noisy2[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            yhn = batch.yhn[:batch.batch_size].squeeze()
            
            loss_1, loss_2, pure_ratio_1, pure_ratio_2, ind_1_update, ind_2_update, ind_noisy_1, ind_noisy_2  = self.criterion(out1, out2, yhn, self.rate_schedule[epoch], batch.n_id, self.noise_or_not)
            
            if epoch > self.config['ct_tk']:
                # Rewire
                new_edge = topk_rewire(batch.x, batch.edge_index, self.device, k_percent=self.config['spl_rewire_rate'])

                hedge_pure1, _, _, hedge_noisy1, _, _ = model1(batch.x, new_edge, noise_rate=self.config['spl_noise_rate_pos'], n_id=batch.n_id)
                hedge_pure2, _, _, hedge_noisy2, _, _ = model2(batch.x, new_edge, noise_rate=self.config['spl_noise_rate_pos'], n_id=batch.n_id)
                #hneg_pure1, _, _, hneg_noisy1, _, _ = model1(batch.x, batch.edge_index, noise_rate=self.config['spl_noise_rate_neg'], n_id=batch.n_id)
                #hneg_pure2, _, _, hneg_noisy2, _, _ = model2(batch.x, batch.edge_index, noise_rate=self.config['spl_noise_rate_neg'], n_id=batch.n_id)
                new_x = shuffle_pos(batch.x, device=self.device, prob=self.config['spl_noise_rate_neg'])
                hneg_pure1, _, _, hneg_noisy1, _, _ = model1(new_x, batch.edge_index, noise_rate=self.config['spl_noise_rate_neg'], n_id=batch.n_id)
                hneg_pure2, _, _, hneg_noisy2, _, _ = model2(new_x, batch.edge_index, noise_rate=self.config['spl_noise_rate_neg'], n_id=batch.n_id)
                # Contrastive
                """
                logits_pa1, logits_pb1, logits_n1 = self.discriminator(h_pure1[ind_noisy_2], hedge_pure1[ind_noisy_2], hneg_noisy1[ind_noisy_2])
                logits_pa2, logits_pb2, logits_n2 = self.discriminator(h_pure2[ind_noisy_1], hedge_pure2[ind_noisy_1], hneg_noisy2[ind_noisy_1])
                loss_cont1 = self.cont_criterion(logits_pa1, logits_pb1, logits_n1)
                loss_cont2 = self.cont_criterion(logits_pa2, logits_pb2, logits_n2)"""
                logits_pa1, logits_n1 = self.discriminator(h_pure1[ind_noisy_1], hedge_pure1[ind_noisy_1], hneg_noisy1[ind_noisy_1])
                logits_pa2, logits_n2 = self.discriminator(h_pure2[ind_noisy_2], hedge_pure2[ind_noisy_2], hneg_noisy2[ind_noisy_2])
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
    
    def evaluate_ct(self, valid_loader, model1, model2):
        model1.eval()
        model2.eval()

        total_correct_1 = 0
        total_correct_2 = 0
        
        for batch in valid_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            _, _, z_pure1, _, _, _ = model1(batch.x, batch.edge_index, n_id=batch.n_id)
            _, _, z_pure2, _, _, _ = model2(batch.x, batch.edge_index, n_id=batch.n_id)
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
            # Only consider predictions and labels of seed nodes
            _, _, z_pure1, _, _, _ = model1(batch.x, batch.edge_index, n_id=batch.n_id)
            _, _, z_pure2, _, _, _ = model2(batch.x, batch.edge_index, n_id=batch.n_id)
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
            
            if self.config['compare_loss'] == 'normal':
                loss = F.cross_entropy(out, yhn)
                #loss = F.cross_entropy(out, y)
            else:
                loss = backward_correction(out, yhn, self.noise_mat, self.config['nbr_classes'], self.device)
            
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
        
        if self.config['do_train']:
            if self.config['train_type'] in ['nalgo','both']:
                print('Train nalgo')
                self.logger.info('Train nalgo')
                #self.model1.network.reset_parameters()
                #self.model2.network.reset_parameters()

                train_loss_1_hist = []
                train_loss_2_hist = []
                train_loss_cont_1_hist = []
                train_loss_cont_2_hist = []
                train_acc_1_hist = []
                train_acc_2_hist = []
                pure_ratio_1_hist = []
                pure_ratio_2_hist = []
                val_acc_1_hist = []
                val_acc_2_hist = []
                test_acc_1_hist = []
                test_acc_2_hist = []

                for epoch in range(self.config['max_epochs']):
                    train_loss_1, train_loss_2, train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list, train_loss_cont_1, train_loss_cont_2 = self.train_ct(self.train_loader, epoch, self.model1.network.to(self.device), self.model2.network.to(self.device), self.optimizer)

                    train_loss_1_hist.append(train_loss_1), train_loss_2_hist.append(train_loss_2)
                    train_loss_cont_1_hist.append(train_loss_cont_1), train_loss_cont_2_hist.append(train_loss_cont_2)
                    train_acc_1_hist.append(train_acc_1), train_acc_2_hist.append(train_acc_2)
                    pure_ratio_1_hist.append(pure_ratio_1_list), pure_ratio_2_hist.append(pure_ratio_2_list)
                    
                    val_acc_1, val_acc_2 = self.evaluate_ct(self.valid_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
                    val_acc_1_hist.append(val_acc_1), val_acc_2_hist.append(val_acc_2)
                    
                    test_acc_1, test_acc_2 = self.test_ct(self.test_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
                    test_acc_1_hist.append(test_acc_1), test_acc_2_hist.append(test_acc_2)
                    self.logger.info('   Train epoch {}/{} --- acc t1: {:.3f} t2: {:.3f} v1: {:.3f} v2: {:.3f} tst1: {:.3f} tst2: {:.3f}'.format(epoch+1,self.config['max_epochs'],train_acc_1,train_acc_2,val_acc_1,val_acc_2,test_acc_1,test_acc_2))
                    """
                    if (test_acc_1 > best_test):
                        best_test = test_acc_1
                    elif (test_acc_2 > best_test):
                        best_test = test_acc_2
                    
                    if (val_acc_1 > best_val):
                        print("saved model, val acc {:.3f}".format(val_acc_1))
                        self.logger.info('   Saved  model')
                        best_val = val_acc_1
                        torch.save(self.model1.network.state_dict(), '../out_model/' + self.config['algo_type'] + '/' + self.output_name + '_m1.pth')
                        torch.save(self.model2.network.state_dict(), '../out_model/' + self.config['algo_type'] + '/' + self.output_name + '_m2.pth')
                    """
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
                    self.logger.info('   Train epoch {}/{} --- acc t: {:.3f} v: {:.3f} tst: {:.3f}'.format(epoch+1,self.config['max_epochs'],train_acc,val_acc,test_acc))

            print('Done training')
            self.logger.info('Done training')
        else:
            print('load')
            self.logger.info('Load trained model')
            self.model1, self.model2 = NGNN(), NGNN()
            self.model1.network.load_state_dict(torch.load('../out_model/coteaching/dt624_id2_both_coteaching_sage_algo_normal_noise_next_pair0.45_lay2_hid128_lr0.001_epo25_bs1024_drop0.5_tk5_colambda0.1_neigh15105_m1.pth'))
            self.model2.network.load_state_dict(torch.load('../out_model/coteaching/dt624_id2_both_coteaching_sage_algo_normal_noise_next_pair0.45_lay2_hid128_lr0.001_epo25_bs1024_drop0.5_tk5_colambda0.1_neigh15105_m2.pth'))

            val_acc_1, val_acc_2 = self.evaluate_ct(self.valid_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
            self.logger.info('   Load eval v1: {:.3f} v2: {:.3f}'.format(val_acc_1,val_acc_2))

        if self.config['train_type'] in ['nalgo','both']:
            self.logger.info('Best test acc1: {:.3f}   acc2: {:.3f}'.format(max(test_acc_1_hist),max(test_acc_2_hist)))
        if self.config['train_type'] in ['baseline','both']:
            self.logger.info('Best baseline test acc: {:.3f}'.format(max(test_acc_hist)))
        
        print('Done')

        if self.config['do_plot']:
            fig, axs = plt.subplots(4, 1, figsize=(10, 15))
            
            #axs[0].axhline(y=0.80, color='grey', linestyle='--')
            #axs[0].axhline(y=0.75, color='grey', linestyle='--')
            if self.config['train_type'] in ['nalgo','both']:
                line1, = axs[0].plot(train_acc_1_hist, 'blue', label="train_acc_1_hist")
                line2, = axs[0].plot(train_acc_2_hist, 'darkgreen', label="train_acc_2_hist")
                line3, = axs[0].plot(val_acc_1_hist, 'purple', label="val_acc_1_hist")
                line4, = axs[0].plot(val_acc_2_hist, 'darkseagreen', label="val_acc_2_hist")
                line5, = axs[1].plot(test_acc_1_hist, 'deepskyblue', label="test_acc_1_hist")
                line6, = axs[1].plot(test_acc_2_hist, 'chartreuse', label="test_acc_2_hist")
                axs[2].plot(pure_ratio_1_hist, 'blue', label="pure_ratio_1_hist")
                axs[2].plot(pure_ratio_2_hist, 'darkgreen', label="pure_ratio_2_hist")
                axs[2].legend()

                axs[3].plot(train_loss_1_hist, 'blue', label="train_loss_1_hist")
                axs[3].plot(train_loss_2_hist, 'darkgreen', label="train_loss_2_hist")
                axs[3].plot(train_loss_cont_1_hist, 'aqua', label="train_loss_cont_1_hist")
                axs[3].plot(train_loss_cont_2_hist, 'lawngreen', label="train_loss_cont_2_hist")
                
            if self.config['train_type'] in ['baseline','both']:
                line7, = axs[0].plot(train_acc_hist, 'red', label="train_acc_hist")
                line8, = axs[0].plot(val_acc_hist, 'tomato', label="val_acc_hist")
                line9, = axs[1].plot(test_acc_hist, 'deeppink', label="test_acc_hist")

                axs[3].plot(train_loss_hist, 'red', label="train_loss_hist")
            
            if self.config['train_type'] in ['nalgo']:
                axs[0].legend(handles=[line1, line2, line3, line4], loc='upper left', bbox_to_anchor=(1.05, 1))
                axs[1].legend(handles=[line5, line6], loc='upper left', bbox_to_anchor=(1.05, 1))
            elif self.config['train_type'] in ['baseline']:
                axs[0].legend(handles=[line7, line8], loc='upper left', bbox_to_anchor=(1.05, 1))
                axs[1].legend(handles=[line9], loc='upper left', bbox_to_anchor=(1.05, 1))
            else:
                axs[0].legend(handles=[line1, line2, line3, line4, line7, line8], loc='upper left', bbox_to_anchor=(1.05, 1))
                axs[1].legend(handles=[line5, line6, line9], loc='upper left', bbox_to_anchor=(1.05, 1))
            
            axs[0].set_title('Plot 1')
            axs[1].set_title('Plot 2')
            axs[2].set_title('Plot 3')
            axs[3].legend()
            axs[3].set_title('Plot 4')

            plt.tight_layout()
            #plt.show()
            plot_name = '../out_plots/coteaching_test/' + self.output_name + '.png'
            plt.savefig(plot_name)

