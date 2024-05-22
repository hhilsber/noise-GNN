import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import Evaluator
import datetime as dt

from .utils.load_utils import load_network
from .utils.data_utils import Jensen_Shannon, Discriminator_innerprod, BCEExeprtLoss
from .utils.augmentation import augment_edges_pos, augment_edges_neg, shuffle_pos, shuffle_neg
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
        self.model1 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'])
        self.model2 = NGNN(self.config['nbr_features'],self.config['hidden_size'],self.config['nbr_classes'],self.config['num_layers'],self.config['dropout'],self.config['learning_rate'],self.config['optimizer'],self.config['module'])
        self.evaluator = Evaluator(name=config['dataset_name'])
        # Coteaching param
        self.criterion = CTLoss(self.device)
        self.rate_schedule = np.ones(self.config['max_epochs'])
        self.rate_schedule[:self.config['ct_tk']] = np.linspace(0, self.config['noise_rate'], self.config['ct_tk'])
        self.rate_schedule[self.config['ct_tk']:self.config['warmup']] = self.rate_schedule[self.config['ct_tk']:self.config['warmup']]*self.config['noise_rate']*self.config['ct_tau']
        #print('rate_schedule: {}'.format(self.rate_schedule))
        # Contrastive
        self.discriminator = Discriminator_innerprod()
        self.cont_criterion = BCEExeprtLoss(self.config['batch_size'])
        
        # Logger and data loader
        date = dt.datetime.date(dt.datetime.now())
        self.output_name = 'dt{}{}_id{}_{}_{}_{}_noise_{}{}_lay{}_hid{}_lr{}_bs{}_drop{}_epo{}_warmup{}_lambda{}_cttk{}_cttau{}'.format(date.month,date.day,self.config['batch_id'],self.config['train_type'],self.config['algo_type'],self.config['module'],self.config['noise_type'],self.config['noise_rate'],self.config['num_layers'],self.config['hidden_size'],self.config['learning_rate'],self.config['batch_size'],self.config['dropout'],self.config['max_epochs'],self.config['warmup'],self.config['lambda'],self.config['ct_tk'],self.config['ct_tau'])
        self.logger = initialize_logger(self.config, self.output_name)
        np.save('../out_nmat/' + self.output_name + '.npy', noise_mat)

        # Graph augmentation
        if config['augment_edge']:
            self.edge_pos = augment_edges_pos(self.data.edge_index, config['nbr_nodes'], config['edge_prob'])
            self.edge_neg = augment_edges_neg(self.data.edge_index, config['nbr_nodes'])
        if config['augment_feat']:
            self.feature_pos = shuffle_pos(self.data.x, config['device'], config['feat_prob'])
            self.feature_neg = shuffle_neg(self.data.x, config['device'])
        print('ok')
        
    
    def warmup(self, epoch, train_loader, model1, optimizer1, model2, optimizer2):
        model1.train()
        model2.train()

        total_loss_1=0
        total_loss_2=0
        total_correct_1=0
        total_correct_2=0
        for batch in train_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out1 = model1(batch.x, batch.edge_index)[0][:batch.batch_size]
            out2 = model2(batch.x, batch.edge_index)[0][:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            yhn = batch.yhn[:batch.batch_size].squeeze()
            
            loss_1, loss_2, pure_ratio_1, pure_ratio_2, _, _, _, _ = self.criterion(out1, out2, yhn, self.rate_schedule[epoch], batch.n_id, self.noise_or_not)
            
            total_loss_1 += float(loss_1)
            total_loss_2 += float(loss_2)
            total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())
            
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
        #self.logger.info('   Warmup epoch {}/{} --- loss1: {:.3f} loss2: {:.3f} acc1: {:.3f} acc2: {:.3f}'.format(epoch+1,self.config['warmup'],train_loss_1,train_loss_2,train_acc_1,train_acc_2))
        return train_loss_1,train_loss_2,train_acc_1,train_acc_2

    def split(self, epoch, train_loader, model1, model2):
        # Pred
        model1.eval()
        model2.eval()
        clean_1 = torch.tensor([])
        clean_2 = torch.tensor([])
        noisy_1 = torch.tensor([])
        noisy_2 = torch.tensor([])
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            with torch.no_grad():
                out1 = torch.nn.Softmax(dim=1)(model1(batch.x, batch.edge_index)[0][:batch.batch_size])
                out2 = torch.nn.Softmax(dim=1)(model2(batch.x, batch.edge_index)[0][:batch.batch_size])
                yhn = batch.yhn[:batch.batch_size].squeeze()

                _, _, _, _, ind_clean_1, ind_clean_2, ind_noisy_1, ind_noisy_2 = self.criterion(out1, out2, yhn, self.rate_schedule[epoch], batch.n_id, self.noise_or_not)
            
            clean_1 = torch.cat((clean_1, ind_clean_1), dim=0)
            clean_2 = torch.cat((clean_2, ind_clean_2), dim=0)
            noisy_1 = torch.cat((noisy_1, ind_noisy_1), dim=0)
            noisy_2 = torch.cat((noisy_2, ind_noisy_2), dim=0)
        return clean_1.long(), clean_2.long(), noisy_1.long(), noisy_2.long()

    def train(self, epoch, train_loader, pos_loader, neg_loader, model, optimizer):
        # Train
        model.train()

        total_loss_semi = 0
        total_loss_cont = 0
        total_loss = 0
        total_correct = 0

        for (batch, batch_p, batch_n) in zip(train_loader, pos_loader, neg_loader):
            #if (batch.batch_size != 512) or (batch_p.batch_size != 512) or (batch_n.batch_size != 512):
            print('bs normal {}, bs pos {}, bs neg {}'.format(batch.batch_size, batch_p.batch_size, batch_n.batch_size))
            batch, batch_p, batch_n = batch.to(self.device), batch_p.to(self.device), batch_n.to(self.device)
            #clean_set = batch.bool_set[:batch.batch_size]
            #noisy_set = ~batch.bool_set[:batch.batch_size]

            # Only consider predictions and labels of seed nodes
            out_semi = model(batch.x, batch.edge_index)[0][:batch.batch_size]
            out_clo = model(batch.x, batch.edge_index)[1][:batch.batch_size]
            out_clp = model(batch_p.x, batch_p.edge_index)[1][:batch_p.batch_size]
            out_cln = model(batch_n.x, batch_n.edge_index)[1][:batch_n.batch_size]
            y = batch.y[:batch.batch_size].squeeze()
            yhn = batch.yhn[:batch.batch_size].squeeze()
            
            # Semi
            loss_semi = F.cross_entropy(out_semi, yhn)
            # Contrastive
            logits_p, logits_n = self.discriminator(out_clo, out_clp, out_cln)
            loss_cont = self.cont_criterion(logits_p, logits_n)
            # Loss
            loss = loss_semi + self.config['lambda'] * loss_cont

            total_loss_semi += float(loss_semi)
            total_loss_cont += float(loss_cont)
            total_loss += float(loss)
            total_correct += int(out_semi.argmax(dim=-1).eq(y).sum())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss_semi = total_loss_semi / len(train_loader)
        total_loss_cont = total_loss_cont / len(train_loader)
        total_loss = total_loss / len(train_loader)
        train_acc = total_correct / self.split_idx['train'].size(0)
        return total_loss_semi, total_loss_cont, total_loss, train_acc
    
    def evaluate(self, valid_loader, model):
        model.eval()
        total_correct = 0
        
        for batch in valid_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out = model(batch.x, batch.edge_index)[0][:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()

            total_correct += int(out.argmax(dim=-1).eq(y).sum())
        val_acc = total_correct / self.split_idx['valid'].size(0)
        return val_acc

    def evaluate_ct(self, valid_loader, model1, model2):
        model1.eval()
        model2.eval()

        total_correct_1 = 0
        total_correct_2 = 0
        
        for batch in valid_loader:
            batch = batch.to(self.device)
            # Only consider predictions and labels of seed nodes
            out1 = model1(batch.x, batch.edge_index)[0][:batch.batch_size]
            out2 = model2(batch.x, batch.edge_index)[0][:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()

            total_correct_1 += int(out1.argmax(dim=-1).eq(y).sum())
            total_correct_2 += int(out2.argmax(dim=-1).eq(y).sum())
        val_acc_1 = total_correct_1 / self.split_idx['valid'].size(0)
        val_acc_2 = total_correct_2 / self.split_idx['valid'].size(0)
        return val_acc_1, val_acc_2

    def create_loaders(self, batch_size, noise=False, clean_idx=None, noise_idx=None):
        print('batch size {}'.format(batch_size))
        train_loader = NeighborLoader(
            self.data,
            input_nodes=self.split_idx['train'],
            num_neighbors=self.config['nbr_neighbors'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config['num_workers'],
            persistent_workers=True
        )
        valid_loader = NeighborLoader(
            self.data,
            input_nodes=self.split_idx['valid'],
            num_neighbors=self.config['nbr_neighbors'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config['num_workers'],
            persistent_workers=True
        )
        if noise:
            pos_loader = NeighborLoader(
                Data(x=self.data.x, y=self.data.y, edge_index=self.edge_pos),
                input_nodes=self.split_idx['train'],
                num_neighbors=self.config['nbr_neighbors'],
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.config['num_workers'],
                persistent_workers=True
            )
            neg_loader = NeighborLoader(
                Data(x=self.feature_neg, y=self.data.y, edge_index=self.edge_neg),
                input_nodes=self.split_idx['train'],
                num_neighbors=self.config['nbr_neighbors'],
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.config['num_workers'],
                persistent_workers=True
            )
        else:
            pos_loader = None
            neg_loader = None
        return train_loader, pos_loader, neg_loader, valid_loader

    def loop(self):
        print('loop')
        
        best_val = 0.3
        # Warmup
        self.logger.info('Warmup')
        train_loader, _, _, valid_loader = self.create_loaders(self.config['batch_size'])
        
        for epoch in range(self.config['warmup']):
            train_loss_1,train_loss_2,train_acc_1,train_acc_2 = self.warmup(epoch, train_loader, self.model1.network.to(self.device), self.model1.optimizer, self.model2.network.to(self.device), self.model2.optimizer)
            val_acc_1, val_acc_2 = self.evaluate_ct(valid_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
            self.logger.info('   Warmup epoch {}/{} --- loss1: {:.3f} loss2: {:.3f} t1: {:.3f} t2: {:.3f} v1: {:.3f} v2: {:.3f}'.format(epoch+1,self.config['warmup'],train_loss_1,train_loss_2,train_acc_1,train_acc_2,val_acc_1,val_acc_2))
            if (val_acc_1 > best_val):
                print("saved model, val acc {:.3f}".format(val_acc_1))
                best_val = val_acc_1
                torch.save(self.model1.network.cpu().state_dict(), '../out_model/' + self.output_name + '_m1.pth')
                torch.save(self.model2.network.cpu().state_dict(), '../out_model/' + self.output_name + '_m2.pth')
        
        # Split data in clean and noisy sets
        print("load")
        self.logger.info('load')
        self.model1, self.model2 = NGNN(), NGNN()
        self.model1.network.load_state_dict(torch.load('../out_model/' + self.output_name + '_m1.pth'))
        self.model2.network.load_state_dict(torch.load('../out_model/' + self.output_name + '_m2.pth'))
        #self.model1.network.load_state_dict(torch.load('../out_model/dt522_id3_contrastive_contrastive_sageFC_noise_next_pair0.45_lay2_hid128_lr0.001_bs1024_drop0.5_epo4_warmup3_lambda1.0_cttk1_cttau1.1_m1.pth'))
        #self.model2.network.load_state_dict(torch.load('../out_model/dt522_id3_contrastive_contrastive_sageFC_noise_next_pair0.45_lay2_hid128_lr0.001_bs1024_drop0.5_epo4_warmup3_lambda1.0_cttk1_cttau1.1_m2.pth'))
        epoch=3
        self.logger.info('Split epoch {}'.format(epoch+1))
        clean_1, clean_2, noisy_1, noisy_2 = self.split(epoch, train_loader, self.model1.network.to(self.device), self.model2.network.to(self.device))
        print(clean_1.shape)
        # Check stats
        clean_ratio1, clean_ratio1_tot = torch.sum(self.noise_or_not[clean_1]).item()/clean_1.shape[0], torch.sum(self.noise_or_not[clean_1]).item()/self.split_idx['train'].size(0)
        noisy_ratio1, noisy_ratio1_tot = torch.sum(self.noise_or_not[noisy_1]).item()/noisy_1.shape[0], torch.sum(self.noise_or_not[noisy_1]).item()/self.split_idx['train'].size(0)
        
        self.logger.info('clean ratio in clean {:.3f}, clean ratio tot {:.3f}'.format(clean_ratio1, clean_ratio1_tot))
        self.logger.info('clean raion in noisy {:.3f}, clean ratio in noisy tot {:.3f}'.format(noisy_ratio1, noisy_ratio1_tot))
        self.logger.info('nbr clean samples {}, noisy samples {}, sum {} == {} total train?'.format(clean_1.shape[0], noisy_1.shape[0], (clean_1.shape[0]+noisy_1.shape[0]), self.split_idx['train'].size(0)))
        """
        # Create clean and noisy loaders
        #bool_set = np.ones_like(self.noise_or_not)
        #bool_set[noisy_1] = 0 # 1 if clean set, 0 if noisy set
        #self.data.bool_set = bool_set
        train_loader, pos_loader, neg_loader, valid_loader = self.create_loaders(batch_size=512, noise=True, clean_idx=clean_1, noise_idx=noisy_1)
        self.logger.info('Train')
        print('len train {} pos {} neg {}'.format(len(train_loader), len(pos_loader), len(neg_loader)))
        
        for epoch in range(self.config['warmup'],self.config['max_epochs']):
            # Train
            total_loss_semi, total_loss_cont, total_loss, train_acc = self.train(epoch, train_loader, pos_loader, neg_loader, self.model1.network.to(self.device), self.model1.optimizer)
            # Eval
            valid_acc = self.evaluate(valid_loader, self.model1.network.to(self.device))
            if not((epoch+1)%1) or ((epoch+1)==self.config['max_epochs']):
                self.logger.info('   Train epoch {}/{} --- loss semi: {:.3f} loss cont: {:.3f} total loss {:.3f} --- train acc: {:.3f} val acc {:.3f}'.format(epoch+1,self.config['max_epochs'],total_loss_semi, total_loss_cont, total_loss, train_acc, valid_acc))
        """
        self.logger.info('Done')