import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, NeighborSampler
import matplotlib.pyplot as plt
from ogb.nodeproppred import Evaluator
import datetime as dt
from torcheval.metrics.functional import multiclass_confusion_matrix

from .utils.load_utils import load_network
from .utils.data_utils import Jensen_Shannon, Discriminator_innerprod, BCEExeprtLoss
from sklearn.metrics import accuracy_score
from .utils.augmentation import *
from .utils.utils import initialize_logger
from .utils.noise import flip_label
from .models.model import NGNN
from .utils.losses import *

class PipelineA(object):
    """
    Processing pipeline
    """
    def __init__(self, config):
        # Set metrics:
        self.device = config['device']

        # Data prep
        self.data, dataset = load_network(config)
        train_idx = self.data.train_mask.nonzero().squeeze()
        val_idx = self.data.val_mask.nonzero().squeeze()
        test_idx = self.data.test_mask.nonzero().squeeze()
      
        self.split_idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        if config['batch_size_full']:
            config['batch_size'] = self.split_idx['train'].shape[0]
           
        print('noise type and rate: {} {}'.format(config['noise_type'], config['noise_rate']))
        print('train: {}, valid: {}, test: {}'.format(self.split_idx['train'].shape[0],self.split_idx['valid'].shape[0],self.split_idx['test'].shape[0]))

        config['nbr_features'] = dataset.num_features #dataset.x.shape[-1]
        config['nbr_classes'] = dataset.num_classes #dataset.y.max().item() + 1
        config['nbr_nodes'] = dataset.x.shape[0]

        # Config
        self.config = config

        # Initialize the model
        self.model1 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'],nbr_nodes=self.config['nbr_nodes'])
        self.model2 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'],nbr_nodes=self.config['nbr_nodes'])
        
        self.criterion = CTLoss(self.device)
        self.rate_schedule = np.ones(self.config['max_epochs'])*self.config['noise_rate']*self.config['ct_tau']
        self.rate_schedule[:self.config['ct_tk']] = np.linspace(0, self.config['noise_rate']*self.config['ct_tau'], self.config['ct_tk'])
        self.optimizer = torch.optim.Adam(list(self.model1.network.parameters()) + list(self.model2.network.parameters()),lr=config['learning_rate'])

        # Contrastive
        self.discriminator = Discriminator_innerprod()
        self.cont_criterion = BCEExeprtLoss(self.config['batch_size'])
        
        # Logger and data loader
        date = dt.datetime.date(dt.datetime.now())
        self.output_name = 'dt{}{}_{}_id{}_{}_{}_{}_noise_{}{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_tk{}_cttau{}_neigh{}{}'.format(date.month,date.day,self.config['dataset_name'],self.config['batch_id'],self.config['train_type'],self.config['algo_type'],self.config['module'],self.config['noise_type'],self.config['noise_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['max_epochs'],self.config['batch_size'],self.config['dropout'],self.config['ct_tk'],self.config['ct_tau'],self.config['nbr_neighbors'][0],self.config['nbr_neighbors'][1])#,self.config['nbr_neighbors'][2])
        self.logger = initialize_logger(self.config, self.output_name)
        #np.save('../out_nmat/' + self.output_name + '.npy', noise_mat)
        
        self.data.yhn, self.noise_mat = flip_label(self.data.y, self.config['nbr_classes'], self.config['noise_type'], self.config['noise_rate'])
        self.noise_or_not = (self.data.y.squeeze() == self.data.yhn) #.int() # true if same lbl
        self.logger.info('   Noise mat {}'.format(self.noise_mat))
        
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
        total_clean1=0
        total_clean2=0
        total_noisy1=0
        total_noisy2=0
        total_ratio_1=0
        total_ratio_2=0

        for batch in train_loader:
            batch = batch.to(self.device)

            # Only consider predictions and labels of seed nodes
            h_pure1, _, z_pure1, _, _, _ = model1(batch.x, batch.edge_index, noise_rate=self.config['spl_noise_rate_pos'], n_id=batch.n_id)
            h_pure2, _, z_pure2, _, _, _ = model2(batch.x, batch.edge_index, noise_rate=self.config['spl_noise_rate_pos'], n_id=batch.n_id)
            
            out1 = z_pure1[:batch.batch_size]
            out2 = z_pure2[:batch.batch_size]
            y = batch.y[:batch.batch_size]
            yhn = batch.yhn[:batch.batch_size].squeeze()
            
            loss_1, loss_2, pure_ratio_1, pure_ratio_2, ind_update_1, ind_update_2, ind_noisy_1, ind_noisy_2  = self.criterion(out1, out2, yhn, self.rate_schedule[epoch], batch.n_id, self.noise_or_not)
            
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
                loss_cont1 = self.cont_criterion(logits_pa1, logits_n1)
                loss_cont2 = self.cont_criterion(logits_pa2, logits_n2)

                loss = loss_1 + loss_2 + self.config['spl_cont_beta'] * loss_cont1 + self.config['spl_cont_beta'] * loss_cont2
            else:
                loss = loss_1 + loss_2
                loss_cont1 = 0
                loss_cont2 = 0
            #pure_ratio_1, pure_ratio_2, ind_update_1, ind_update_2, ind_noisy_1, ind_noisy_2
            y_true = y.cpu()
            y_pred1 = out1.argmax(dim=-1, keepdim=True).cpu()
            y_pred2 = out2.argmax(dim=-1, keepdim=True).cpu()
            ind_update_1, ind_update_2 = ind_update_1.cpu(), ind_update_2.cpu()
            ind_noisy_1, ind_noisy_2 = ind_noisy_1.cpu(), ind_noisy_2.cpu()
            """
            correct_clean1 += int(y_pred1[ind_update_1].eq(y[ind_update_1]).sum())
            correct_clean2 += int(y_pred2[ind_update_2].eq(y[ind_update_2]).sum())
            correct_noisy1 += int(y_pred1[ind_noisy_1].eq(y[ind_noisy_1]).sum())
            correct_noisy2 += int(y_pred2[ind_noisy_2].eq(y[ind_noisy_2]).sum())"""
            total_clean1 += accuracy_score(y_pred1[ind_update_1], y_true[ind_update_1])
            total_clean2 += accuracy_score(y_pred2[ind_update_2], y_true[ind_update_2])
            if epoch > 0:
                total_noisy1 += accuracy_score(y_pred1[ind_noisy_1], y_true[ind_noisy_1])
                total_noisy2 += accuracy_score(y_pred2[ind_noisy_2], y_true[ind_noisy_2])
            else:
                total_noisy1 += 0
                total_noisy2 += 0
            #print("clean 1 {} noisy 1{}".format(correct_clean1,correct_noisy1))

            total_loss_1 += float(loss_1)
            total_loss_2 += float(loss_2)
            total_loss_cont_1 += float(loss_cont1)
            total_loss_cont_2 += float(loss_cont2)
            #total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            #total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())
            total_ratio_1 += (100*pure_ratio_1)
            total_ratio_2 += (100*pure_ratio_2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss_1 = total_loss_1 / len(train_loader)
        train_loss_2 = total_loss_2 / len(train_loader)
        train_loss_cont_1 = total_loss_cont_1 / len(train_loader)
        train_loss_cont_2 = total_loss_cont_2 / len(train_loader)
        #train_acc_1 = total_correct_1 / self.split_idx['train'].size(0)
        #train_acc_2 = total_correct_2 / self.split_idx['train'].size(0)
        #acc_clean = total_acc_clean / len(train_loader)
        #acc_noisy = total_acc_noisy / len(train_loader)
        correct_clean1 = total_clean1 / len(train_loader)
        correct_clean2 = total_clean2 / len(train_loader)
        correct_noisy1 = total_noisy1 / len(train_loader)
        correct_noisy2 = total_noisy2 / len(train_loader)
        pure_ratio_1_list = total_ratio_1 / len(train_loader)
        pure_ratio_2_list = total_ratio_2 / len(train_loader)
        
        return train_loss_1, train_loss_2, pure_ratio_1_list, pure_ratio_2_list, train_loss_cont_1, train_loss_cont_2, correct_clean1, correct_clean2, correct_noisy1, correct_noisy2
    

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
    
    def eval(self, subgraph_loader, model):
        model.eval()

        with torch.no_grad():
            out = model.inference(self.data.x, subgraph_loader, self.device)
            
            y_true = self.data.y.cpu()
            y_pred = out.argmax(dim=-1, keepdim=True).squeeze()
            print(y_true[:20])
            print(y_pred[:20])
            #train_acc = accuracy_score(y_true[self.split_idx['train']], y_pred[self.split_idx['train']])
            #val_acc = accuracy_score(y_true[self.split_idx['valid']], y_pred[self.split_idx['valid']])
            #test_acc = accuracy_score(y_true[self.split_idx['test']], y_pred[self.split_idx['test']])
            cm = multiclass_confusion_matrix(y_pred[self.split_idx['test']], y_true[self.split_idx['test']], self.config['nbr_classes'])
            print(cm)
            normalized_tensor = cm / cm.sum(dim=1, keepdim=True)
            print(normalized_tensor)
        return cm

    def loop(self):
        print('loop')
        
        if self.config['do_train']:
            best_acc_ct = []
            self.logger.info('{} RUNS'.format(self.config['num_runs']))
            if self.config['train_type'] in ['nalgo','both']:
                for i in range(self.config['num_runs']):
                    best_val = 0
                    #self.logger.info('   Train nalgo')
                    self.model1.network.reset_parameters()
                    self.model2.network.reset_parameters()

                    train_loss_1_hist = []
                    train_loss_2_hist = []
                    train_loss_cont_1_hist = []
                    train_loss_cont_2_hist = []
                    clean_hist1 = []
                    clean_hist2 = []
                    noisy_hist1 = []
                    noisy_hist2 = []
                    pure_ratio_1_hist = []
                    pure_ratio_2_hist = []
                    train_acc_1_hist = []
                    train_acc_2_hist = []
                    val_acc_1_hist = []
                    val_acc_2_hist = []
                    test_acc_1_hist = []
                    test_acc_2_hist = []

                    for epoch in range(self.config['max_epochs']):
                        train_loss_1, train_loss_2, pure_ratio_1_list, pure_ratio_2_list, train_loss_cont_1, train_loss_cont_2, correct_clean1, correct_clean2, correct_noisy1, correct_noisy2 = self.train_ct(self.train_loader, epoch, self.model1.network.to(self.device), self.model2.network.to(self.device), self.optimizer)
                        train_acc_1, val_acc_1, test_acc_1 = self.test_planet(self.subgraph_loader, self.model1.network.to(self.device))
                        train_acc_2, val_acc_2, test_acc_2 = self.test_planet(self.subgraph_loader, self.model2.network.to(self.device))

                        train_loss_1_hist.append(train_loss_1), train_loss_2_hist.append(train_loss_2)
                        train_loss_cont_1_hist.append(train_loss_cont_1), train_loss_cont_2_hist.append(train_loss_cont_2)
                        train_acc_1_hist.append(train_acc_1), train_acc_2_hist.append(train_acc_2)

                        clean_hist1.append(correct_clean1), clean_hist2.append(correct_clean2)
                        noisy_hist1.append(correct_noisy1), noisy_hist2.append(correct_noisy2)
                        pure_ratio_1_hist.append(pure_ratio_1_list), pure_ratio_2_hist.append(pure_ratio_2_list)
                        
                        #val_acc_1, val_acc_2 = self.evaluate_ct(self.valid_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
                        val_acc_1_hist.append(val_acc_1), val_acc_2_hist.append(val_acc_2)
                        
                        #test_acc_1, test_acc_2 = self.test_ct(self.test_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
                        test_acc_1_hist.append(test_acc_1), test_acc_2_hist.append(test_acc_2)
                        if self.config['epoch_logger']:
                            self.logger.info('   Train epoch {}/{} --- acc t1: {:.3f} t2: {:.3f} v1: {:.3f} v2: {:.3f} tst1: {:.3f} tst2: {:.3f}'.format(epoch+1,self.config['max_epochs'],train_acc_1,train_acc_2,val_acc_1,val_acc_2,test_acc_1,test_acc_2))
                        
                        if (test_acc_1 > best_val):
                            print("saved model 1, val acc {:.3f}".format(val_acc_1))
                            self.logger.info('   Saved  model 1')
                            best_val = test_acc_1
                            torch.save(self.model1.network.state_dict(), '../out_model/' + self.config['algo_type'] + '/' + self.output_name + '_model.pth')
                        elif (test_acc_2 > best_val):
                            print("saved model 2, val acc {:.3f}".format(val_acc_1))
                            self.logger.info('   Saved  model 2')
                            best_val = test_acc_2
                            torch.save(self.model2.network.state_dict(), '../out_model/' + self.config['algo_type'] + '/' + self.output_name + '_model.pth')
                    self.logger.info('   RUN {} - best nalgo test acc1: {:.3f}   acc2: {:.3f}'.format(i+1,max(test_acc_1_hist),max(test_acc_2_hist)))
                    best_acc_ct.append(max(max(test_acc_1_hist),max(test_acc_2_hist)))

                std, mean = torch.std_mean(torch.as_tensor(best_acc_ct))
                self.logger.info('   RUN nalgo mean {:.3f} +- {:.3f} std'.format(mean,std))
            
            print('Done training')
            self.logger.info('Done training')
        #else:
        print('load')
        self.logger.info('Load trained model')
        self.model = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'],nbr_nodes=self.config['nbr_nodes'])
        model_name = '../out_model/analysis/dt827_computers_id1_nalgo_analysis_sagePL_noise_rand_pair0.4_lay2_hid512_lr0.001_epo150_bs300_drop0.5_tk25_cttau0.1_neigh105_model.pth'
        self.model.network.load_state_dict(torch.load(model_name))
        
        _ = self.eval(self.subgraph_loader, self.model.network.to(self.device))
        train_acc, val_acc, test_acc = self.test_planet(self.subgraph_loader, self.model.network.to(self.device))
        self.logger.info('   Model loaded {}'.format(model_name))
        self.logger.info('   --- acc t1: {:.3f} v1: {:.3f} tst1: {:.3f}'.format(train_acc,val_acc,test_acc))

        if self.config['do_plot']:
            fig, axs = plt.subplots(4, 1, figsize=(10, 15))
          
            line1, = axs[0].plot(train_acc_1_hist, 'blue', label="train_acc_1_hist")
            line2, = axs[0].plot(train_acc_2_hist, 'darkgreen', label="train_acc_2_hist")
            line3, = axs[0].plot(val_acc_1_hist, 'purple', label="val_acc_1_hist")
            line4, = axs[0].plot(val_acc_2_hist, 'darkseagreen', label="val_acc_2_hist")
            line5, = axs[0].plot(test_acc_1_hist, 'deepskyblue', label="test_acc_1_hist")
            line6, = axs[0].plot(test_acc_2_hist, 'chartreuse', label="test_acc_2_hist")
            line7, = axs[1].plot(clean_hist1, 'pink', label="clean_hist1")
            line8, = axs[1].plot(clean_hist2, 'red', label="clean_hist2")
            line9, = axs[1].plot(noisy_hist1, 'cyan', label="noisy_hist1")
            line10, = axs[1].plot(noisy_hist2, 'blue', label="noisy_hist2")
            axs[2].plot(pure_ratio_1_hist, 'blue', label="pure_ratio_1_hist")
            axs[2].plot(pure_ratio_2_hist, 'darkgreen', label="pure_ratio_2_hist")
            axs[2].legend()

            axs[3].plot(train_loss_1_hist, 'blue', label="train_loss_1_hist")
            axs[3].plot(train_loss_2_hist, 'darkgreen', label="train_loss_2_hist")
            axs[3].plot(train_loss_cont_1_hist, 'aqua', label="train_loss_cont_1_hist")
            axs[3].plot(train_loss_cont_2_hist, 'lawngreen', label="train_loss_cont_2_hist")
            
            axs[0].legend(handles=[line1, line2, line3, line4,line5, line6], loc='upper left', bbox_to_anchor=(1.05, 1))
            axs[1].legend(handles=[line7, line8, line9, line10], loc='upper left', bbox_to_anchor=(1.05, 1))
       
            axs[0].set_title('Plot 1')
            axs[1].set_title('Plot 2')
            axs[2].set_title('Plot 3')
            axs[3].legend()
            axs[3].set_title('Plot 4')

            plt.tight_layout()
            #plt.show()
            plot_name = '../out_analysis/' + self.output_name + '.png'
            plt.savefig(plot_name)