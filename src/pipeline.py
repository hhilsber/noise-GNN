import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, NeighborSampler
import matplotlib.pyplot as plt
from ogb.nodeproppred import Evaluator
from sklearn.metrics import accuracy_score
import datetime as dt

from .utils.load_utils import load_network
from .utils.data_utils import topk_accuracy
from .utils.utils import initialize_logger
from .utils.noise import flip_label
from .models.model import NGNN
from .utils.losses import *

class PipelineCO(object):
    """
    Processing pipeline
    """
    def __init__(self, config):
        # Set metrics:
        self.device = config['device']

        # Data prep
        self.data, dataset = load_network(config)
        print('noise type and rate: {} {}'.format(config['noise_type'], config['noise_rate']))
        #self.data.yhn, self.noise_mat = flip_label(self.data.y, dataset.num_classes, config['noise_type'], config['noise_rate'])
        #self.noise_or_not = (self.data.y.squeeze() == self.data.yhn) #.int() # true if same lbl
        
        config['nbr_features'] = dataset.num_features #self.dataset.x.shape[-1]
        config['nbr_classes'] = dataset.num_classes #dataset.y.max().item() + 1
        config['nbr_nodes'] = dataset.x.shape[0]

        # Config
        self.config = config

        # Initialize the model
        if self.config['train_type'] in ['nalgo','both']:
            #self.model1 = NGNN(config)
            #self.model2 = NGNN(config)
            self.model1 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'])
            self.model2 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'])
            
            self.criterion = CTLoss(self.device)
            self.rate_schedule = np.ones(self.config['max_epochs'])*self.config['noise_rate']*self.config['ct_tau']
            self.rate_schedule[:self.config['ct_tk']] = np.linspace(0, self.config['noise_rate']**self.config['ct_exp'], self.config['ct_tk'])

        if self.config['train_type'] in ['baseline','both']:
            self.model_c = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'])
        
        self.evaluator = Evaluator(name=config['dataset_name'])
        config['whole_test_set'] = True
        if (not config['whole_test_set']) and (config['dataset_name'] in ['ogbn-products']):
            print('not whole test set, whole_test_set = {}'.format(config['whole_test_set']))
            original_split = dataset.get_idx_split()
            self.split_idx = {'train': self.data.new_train_idx, 'valid': self.data.new_valid_idx, 'test': self.data.new_test_idx}
        else:
            self.split_idx = dataset.get_idx_split()
        input_nodes = torch.cat((self.split_idx['train'],self.split_idx['valid'],self.split_idx['test']),dim=0)
        print('input_nodes: {}'.format(input_nodes.shape))
        print('train: {}, valid: {}, test: {}'.format(self.split_idx['train'].shape[0],self.split_idx['valid'].shape[0],self.split_idx['test'].shape[0]))
 
        # Logger and data loader
        date = dt.datetime.date(dt.datetime.now())
        self.output_name = 'dt{}{}_{}_id{}_{}_{}_{}_noise_{}{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_tk{}_cttau{}_neigh{}{}'.format(date.month,date.day,self.config['dataset_name'],self.config['batch_id'],self.config['train_type'],self.config['algo_type'],self.config['module'],self.config['noise_type'],self.config['noise_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['max_epochs'],self.config['batch_size'],self.config['dropout'],self.config['ct_tk'],self.config['ct_tau'],self.config['nbr_neighbors'][0],self.config['nbr_neighbors'][1])
        self.logger = initialize_logger(self.config, self.output_name)
        #np.save('../out_nmat/' + self.output_name + '.npy', noise_mat)
        
        self.data.yhn, self.noise_mat = flip_label(self.data.y, self.config['nbr_classes'], self.config['noise_type'], self.config['noise_rate'])
        self.noise_or_not = (self.data.y.squeeze() == self.data.yhn) #.int() # true if same lbl

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
            input_nodes=input_nodes,
            num_neighbors=self.config['nbr_neighbors'],
            batch_size=4092,
            num_workers=4,
            persistent_workers=True,
        )
        #print('length train_loader: {}, subgraph_loader: {}'.format(len(self.train_loader),len(self.subgraph_loader)))

    def train_ct(self, train_loader, epoch, model1, optimizer1, model2, optimizer2):
        if not((epoch+1)%10) or ((epoch+1)==1):
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
            #print("{}, {}, {}".format(out1.shape,out2.shape,yhn.shape))
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
        train_acc_1 = total_correct_1 / self.split_idx['train'].size(0)
        train_acc_2 = total_correct_2 / self.split_idx['train'].size(0)
        pure_ratio_1_list = total_ratio_1 / len(train_loader)
        pure_ratio_2_list = total_ratio_2 / len(train_loader)
        
        return train_loss_1, train_loss_2, train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list

    def train(self, train_loader, epoch, model, optimizer):
        if not((epoch+1)%10) or ((epoch+1)==1):
            print('   Train epoch {}/{}'.format(epoch+1, self.config['max_epochs']))
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

    def test_ogb(self, subgraph_loader, model):
        model.eval()

        with torch.no_grad():
            out = model.inference(self.data.x, subgraph_loader, self.device)
            
            y_true = self.data.y.cpu()
            y_pred = out.argmax(dim=-1, keepdim=True)

            train_acc = self.evaluator.eval({
                'y_true': y_true[self.split_idx['train']],
                'y_pred': y_pred[self.split_idx['train']],
            })['acc']
            val_acc = self.evaluator.eval({
                'y_true': y_true[self.split_idx['valid']],
                'y_pred': y_pred[self.split_idx['valid']],
            })['acc']
            test_acc = self.evaluator.eval({
                'y_true': y_true[self.split_idx['test']],
                'y_pred': y_pred[self.split_idx['test']],
            })['acc']

        return train_acc, val_acc, test_acc
    

    def loop(self):
        print('loop')
        
        if self.config['do_train']:
            self.logger.info('{} RUNS'.format(self.config['num_runs']))
            if self.config['train_type'] in ['nalgo','both']:
                best_acc_ct = []
                for i in range(self.config['num_runs']):
                    self.data.yhn, self.noise_mat = flip_label(self.data.y, self.config['nbr_classes'], self.config['noise_type'], self.config['noise_rate'])
                    self.noise_or_not = (self.data.y.squeeze() == self.data.yhn) #.int() # true if same lbl

                    self.train_loader = NeighborLoader(
                        self.data,
                        input_nodes=self.split_idx['train'],
                        num_neighbors=self.config['nbr_neighbors'],
                        batch_size=self.config['batch_size'],
                        shuffle=True,
                        num_workers=1,
                        persistent_workers=True
                    )
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

                    best_test = 0.3
                    for epoch in range(self.config['max_epochs']):
                        #train_loss_1, train_loss_2, train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list = self.train_ct(self.train_loader, epoch, self.model1.network.to(self.device), self.model1.optimizer, self.model2.network.to(self.device), self.model2.optimizer)
                        train_loss_1, train_loss_2, _, _, pure_ratio_1_list, pure_ratio_2_list = self.train_ct(self.train_loader, epoch, self.model1.network.to(self.device), self.model1.optimizer, self.model2.network.to(self.device), self.model2.optimizer)
                    
                        train_acc_1, val_acc_1, test_acc_1 = self.test_ogb(self.subgraph_loader, self.model1.network.to(self.device))
                        train_acc_2, val_acc_2, test_acc_2 = self.test_ogb(self.subgraph_loader, self.model2.network.to(self.device))
                        
                        train_loss_1_hist.append(train_loss_1), train_loss_2_hist.append(train_loss_2)
                        train_acc_1_hist.append(train_acc_1), train_acc_2_hist.append(train_acc_2)
                        pure_ratio_1_hist.append(pure_ratio_1_list), pure_ratio_2_hist.append(pure_ratio_2_list)
                        
                        #val_acc_1, val_acc_2 = self.evaluate_ct(self.valid_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
                        val_acc_1_hist.append(val_acc_1), val_acc_2_hist.append(val_acc_2)
                        
                        #test_acc_1, test_acc_2 = self.test_ct(self.test_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
                        test_acc_1_hist.append(test_acc_1), test_acc_2_hist.append(test_acc_2)
                        if not((epoch+1)%10) and self.config['epoch_logger']:
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
                    self.logger.info('   RUN {} - best nalgo test acc1: {:.3f}   acc2: {:.3f}'.format(i+1,max(test_acc_1_hist),max(test_acc_2_hist)))
                    best_acc_ct.append(max(max(test_acc_1_hist),max(test_acc_2_hist)))
                
                std, mean = torch.std_mean(torch.as_tensor(best_acc_ct))
                self.logger.info('   RUN nalgo mean {:.3f} +- {:.3f} std'.format(mean,std))
                # Save ratio
                ratio_name = '../out_analysis/' + self.output_name + '_ratio.pt'
                print(pure_ratio_1_hist)
                torch.save(torch.FloatTensor(pure_ratio_1_hist), ratio_name)
            if self.config['train_type'] in ['baseline','both']:
                best_acc_bs = []
                for i in range(self.config['num_runs']):
                    
                    
                    #self.logger.info('   Train baseline')
                    self.model_c.network.reset_parameters()

                    train_loss_hist = []
                    train_acc_hist = []
                    val_acc_hist = []
                    test_acc_hist = []
                    for epoch in range(self.config['max_epochs']):
                        #train_loss, train_acc = self.train(self.train_loader, epoch, self.model_c.network.to(self.device), self.model_c.optimizer)
                        train_loss, _ = self.train(self.train_loader, epoch, self.model_c.network.to(self.device), self.model_c.optimizer)
                        train_acc, val_acc, test_acc = self.test_ogb(self.subgraph_loader, self.model_c.network.to(self.device))
                        
                        train_loss_hist.append(train_loss)
                        train_acc_hist.append(train_acc)
                        val_acc_hist.append(val_acc)
                        test_acc_hist.append(test_acc)
                        
                        if not((epoch+1)%10) and self.config['epoch_logger']:
                            self.logger.info('   Train epoch {}/{} --- acc t: {:.3f} v: {:.3f} tst: {:.3f}'.format(epoch+1,self.config['max_epochs'],train_acc,val_acc,test_acc))
                    self.logger.info('   RUN {} - best baseline test acc: {:.3f}'.format(i+1,max(test_acc_hist)))
                    best_acc_bs.append(max(test_acc_hist))
                    
                std, mean = torch.std_mean(torch.as_tensor(best_acc_bs))
                self.logger.info('   RUN baseline mean {:.3f} +- {:.3f} std'.format(mean,std))
            
            print('Done training')
            self.logger.info('Done training')
        else:
            print('load')
            self.logger.info('Load trained model')
            self.model1, self.model2 = NGNN(), NGNN()
            self.model1.network.load_state_dict(torch.load('../out_model/coteaching/dt624_id2_both_coteaching_sage_algo_normal_noise_next_pair0.45_lay2_hid128_lr0.001_epo25_bs1024_drop0.5_tk5_colambda0.1_neigh15105_m1.pth'))
            self.model2.network.load_state_dict(torch.load('../out_model/coteaching/dt624_id2_both_coteaching_sage_algo_normal_noise_next_pair0.45_lay2_hid128_lr0.001_epo25_bs1024_drop0.5_tk5_colambda0.1_neigh15105_m2.pth'))

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
            plot_name = '../out_plots/' + self.config['algo_type'] + self.config['what'] + '/' + self.output_name + '.png'
            plt.savefig(plot_name)

