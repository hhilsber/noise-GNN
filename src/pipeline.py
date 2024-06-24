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
        self.dataset = load_network(config)
        self.split_idx = self.dataset.get_idx_split()
        self.data = self.dataset[0]
        print('noise type and rate: {} {}'.format(config['noise_type'], config['noise_rate']))
        self.data.yhn, noise_mat = flip_label(self.data.y, self.dataset.num_classes, config['noise_type'], config['noise_rate'])
        self.noise_or_not = (self.data.y.squeeze() == self.data.yhn) #.int() # true if same lbl
        
        config['nbr_features'] = self.dataset.num_features #self.dataset.x.shape[-1]
        config['nbr_classes'] = self.dataset.num_classes #dataset.y.max().item() + 1
        config['nbr_nodes'] = self.dataset.x.shape[0]
        
        #train_noise_idx = self.noise_or_not[self.split_idx['train']]
        #valid_noise_idx = self.noise_or_not[self.split_idx['valid']]
        #self.train_wo_noise = self.split_idx['train'][train_noise_idx]
        #valid_wo_noise = self.split_idx['valid'][valid_noise_idx]
        #config['train_wo_noise'] = self.train_wo_noise.size(0)
        #config['train_with_noise'] = self.split_idx['train'].shape[0]

        # Config
        self.config = config

        # Initialize the model
        if self.config['train_type'] in ['nalgo','both']:
            #self.model1 = NGNN(config)
            #self.model2 = NGNN(config)
            self.model1 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'])
            self.model2 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'])
            if self.config['algo_type'] == 'coteaching':
                self.criterion = CTLoss(self.device)
            elif self.config['algo_type'] == 'codi':
                self.criterion = CoDiLoss(self.device, self.config['co_lambda'])
            self.rate_schedule = np.ones(self.config['max_epochs'])*self.config['noise_rate']*self.config['ct_tau']
            self.rate_schedule[:self.config['ct_tk']] = np.linspace(0, self.config['noise_rate']**self.config['ct_exp'], self.config['ct_tk'])

        if self.config['train_type'] in ['baseline','both']:
            self.model_c = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'])
        self.evaluator = Evaluator(name=config['dataset_name'])
        
        print('train: {}, valid: {}, test: {}'.format(self.split_idx['train'].shape[0],self.split_idx['valid'].shape[0],self.split_idx['test'].shape[0]))
        
        # Logger and data loader
        date = dt.datetime.date(dt.datetime.now())
        self.output_name = 'dt{}{}_id{}_{}_{}_{}_algo_{}_noise_{}{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_tk{}_colambda{}_neigh{}{}{}'.format(date.month,date.day,self.config['batch_id'],self.config['train_type'],self.config['algo_type'],self.config['module'],self.config['compare_loss'],self.config['noise_type'],self.config['noise_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['max_epochs'],self.config['batch_size'],self.config['dropout'],self.config['ct_tk'],self.config['co_lambda'],self.config['nbr_neighbors'][0],self.config['nbr_neighbors'][1],self.config['nbr_neighbors'][2])
        self.logger = initialize_logger(self.config, self.output_name)
        np.save('../out_nmat/' + self.output_name + '.npy', noise_mat)

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
            input_nodes=self.split_idx['test'][:int(len(self.split_idx['test'])/30)],
            num_neighbors=self.config['nbr_neighbors'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            persistent_workers=True
        )
        print(len(self.test_loader))
        self.subgraph_loader = NeighborLoader(
            self.data,
            input_nodes=None,
            num_neighbors=self.config['nbr_neighbors'],
            batch_size=32,
            num_workers=self.config['num_workers'],
            persistent_workers=True
        )

    def train_ct(self, train_loader, epoch, model1, optimizer1, model2, optimizer2):
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
        total_ratio_1=0
        total_ratio_2=0

        for batch in train_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out1 = model1(batch.x, batch.edge_index)[:batch.batch_size]
            out2 = model2(batch.x, batch.edge_index)[:batch.batch_size]
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
        
        return train_loss_1, train_loss_2, train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list
    
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
                loss = backward_correction(out, yhn, self.noise_mat, self.device, self.config['nbr_classes'])
            
            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(y).sum()) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / self.split_idx['train'].size(0)
        #train_acc = total_correct / self.train_wo_noise.size(0)
        return train_loss, train_acc

    def evaluate_ct(self, valid_loader, model1, model2):
        model1.eval()
        model2.eval()

        total_correct_1 = 0
        total_correct_2 = 0
        
        for batch in valid_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out1 = model1(batch.x, batch.edge_index)[:batch.batch_size]
            out2 = model2(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()

            total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())
        val_acc_1 = total_correct_1 / self.split_idx['valid'].size(0)
        val_acc_2 = total_correct_2 / self.split_idx['valid'].size(0)
        return val_acc_1, val_acc_2
    
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

    def test(self, model, data, split_idx):
        model.eval()

        out = model.inference(data.x, self.subgraph_loader, self.device)

        y_true = data.y.cpu()
        y_pred = out.argmax(dim=-1, keepdim=True)

        train_acc = evaluator.eval({
            'y_true': y_true[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        val_acc = evaluator.eval({
            'y_true': y_true[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y_true[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })['acc']

        return train_acc, val_acc, test_acc

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
                train_acc_1_hist = []
                train_acc_2_hist = []
                pure_ratio_1_hist = []
                pure_ratio_2_hist = []
                val_acc_1_hist = []
                val_acc_2_hist = []

                best_val = 0.3
                for epoch in range(self.config['max_epochs']):
                    train_loss_1, train_loss_2, train_acc_1, train_acc_2, pure_ratio_1_list, pure_ratio_2_list = self.train_ct(self.train_loader, epoch, self.model1.network.to(self.device), self.model1.optimizer, self.model2.network.to(self.device), self.model2.optimizer)
                    print('train loss {:.3f} --- {:.3f}'.format(train_loss_1,train_loss_2))

                    train_loss_1_hist.append(train_loss_1)
                    train_loss_2_hist.append(train_loss_2)
                    train_acc_1_hist.append(train_acc_1)
                    train_acc_2_hist.append(train_acc_2)
                    pure_ratio_1_hist.append(pure_ratio_1_list)
                    pure_ratio_2_hist.append(pure_ratio_2_list)
                    
                    val_acc_1, val_acc_2 = self.evaluate_ct(self.valid_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
                    val_acc_1_hist.append(val_acc_1)
                    val_acc_2_hist.append(val_acc_2)
                    self.logger.info('   Train epoch {}/{} --- acc t1: {:.4f} t2: {:.4f} v1: {:.4f} v2: {:.4f}'.format(epoch+1,self.config['max_epochs'],train_acc_1,train_acc_2,val_acc_1,val_acc_2))

                    if (val_acc_1 > best_val):
                        print("saved model, val acc {:.3f}".format(val_acc_1))
                        self.logger.info('   Saved  model')
                        best_val = val_acc_1
                        torch.save(self.model1.network.state_dict(), '../out_model/' + self.config['algo_type'] + '/' + self.output_name + '_m1.pth')
                        torch.save(self.model2.network.state_dict(), '../out_model/' + self.config['algo_type'] + '/' + self.output_name + '_m2.pth')
        
            if self.config['train_type'] in ['baseline','both']:
                print('Train baseline')
                self.logger.info('Train baseline')
                #self.model_c.network.reset_parameters()

                train_loss_hist = []
                train_acc_hist = []
                val_acc_hist = []
                for epoch in range(self.config['max_epochs']):
                    train_loss, train_acc = self.train(self.train_loader, epoch, self.model_c.network.to(self.device), self.model_c.optimizer)
                    train_loss_hist.append(train_loss)
                    train_acc_hist.append(train_acc)

                    val_acc = self.evaluate(self.valid_loader, self.model_c.network.to(self.device))
                    val_acc_hist.append(val_acc)
                    self.logger.info('   Train epoch {}/{} --- acc t: {:.4f} v: {:.4f}'.format(epoch+1,self.config['max_epochs'],train_acc,val_acc))
            print('Done training')
            self.logger.info('Done training')
        else:
            print('load')
            self.logger.info('Load warmup model')
            self.model1, self.model2 = NGNN(), NGNN()
            self.model1.network.load_state_dict(torch.load('../out_model/coteaching/dt624_id2_both_coteaching_sage_algo_normal_noise_next_pair0.45_lay2_hid128_lr0.001_epo25_bs1024_drop0.5_tk5_colambda0.1_neigh15105_m1.pth'))
            self.model2.network.load_state_dict(torch.load('../out_model/coteaching/dt624_id2_both_coteaching_sage_algo_normal_noise_next_pair0.45_lay2_hid128_lr0.001_epo25_bs1024_drop0.5_tk5_colambda0.1_neigh15105_m2.pth'))
        
        print('Testing')
        self.logger.info('Test')
        #_, _, _ = self.test(self.model1.network.to(self.device), self.data, self.split_idx)
        test_acc_1, test_acc_2 = self.evaluate_ct(self.test_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
        self.logger.info('   Test acc 1: {:.4f} 2: {:.4f}'.format(test_acc_1,test_acc_2))
        print('Done testing')
        self.logger.info('Done testing')

        if self.config['do_plot']:
            fig, axs = plt.subplots(3, 1, figsize=(10, 15))
            
            axs[0].axhline(y=0.8, color='grey', linestyle='--')
            axs[0].axhline(y=0.9, color='grey', linestyle='--')
            if self.config['train_type'] in ['nalgo','both']:
                line1, = axs[0].plot(train_acc_1_hist, 'blue', label="train_acc_1_hist")
                line2, = axs[0].plot(train_acc_2_hist, 'darkgreen', label="train_acc_2_hist")
                line3, = axs[0].plot(val_acc_1_hist, 'purple', label="val_acc_1_hist")
                line4, = axs[0].plot(val_acc_2_hist, 'chartreuse', label="val_acc_2_hist")
                
                axs[1].plot(pure_ratio_1_hist, 'blue', label="pure_ratio_1_hist")
                axs[1].plot(pure_ratio_2_hist, 'darkgreen', label="pure_ratio_2_hist")
                axs[1].legend()

                axs[2].plot(train_loss_1_hist, 'blue', label="train_loss_1_hist")
                axs[2].plot(train_loss_2_hist, 'darkgreen', label="train_loss_2_hist")
                
            if self.config['train_type'] in ['baseline','both']:
                line5, = axs[0].plot(train_acc_hist, 'red', label="train_acc_hist")
                line6, = axs[0].plot(val_acc_hist, 'peachpuff', label="val_acc_hist")

                axs[2].plot(train_loss_hist, 'red', label="train_loss_hist")
            
            if self.config['train_type'] in ['nalgo']:
                axs[0].legend(handles=[line1, line2, line3, line4], loc='upper left', bbox_to_anchor=(1.05, 1))
            elif self.config['train_type'] in ['baseline']:
                axs[0].legend(handles=[line6, line5], loc='upper left', bbox_to_anchor=(1.05, 1))
            else:
                axs[0].legend(handles=[line1, line2, line3, line4, line5, line6], loc='upper left', bbox_to_anchor=(1.05, 1))
            
            axs[0].set_title('Plot 1')
            axs[1].set_title('Plot 2')
            axs[2].legend()
            axs[2].set_title('Plot 3')

            plt.tight_layout()
            #plt.show()
            plot_name = '../out_plots/coteaching/' + self.output_name + '.png'
            plt.savefig(plot_name)
