import os
import sys

import pickle as pkl
import torch
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, CitationFull, Amazon
import torch_geometric.transforms as T
import datetime as dt



def load_network(config):
    """
    txt
    """

    device = config['device']
    dataset_name = config['dataset_name']
    root = config['data_dir']


    if dataset_name in ['ogbn-products']:
        dataset = PygNodePropPredDataset(dataset_name, root)
        data = dataset[0]
        data = ogb_products_splits(data, dataset.get_idx_split(), dataset.num_classes, config)
    elif dataset_name in ['ogbn-arxiv']:
        #dataset = PygNodePropPredDataset(dataset_name, root, transform=T.ToSparseTensor())
        #dataset = PygNodePropPredDataset(dataset_name, root, T.Compose([T.ToUndirected(),T.ToSparseTensor()]))
        transforms = T.ToUndirected()
        dataset = PygNodePropPredDataset(name = dataset_name, transform=transforms, root = root)
        data = dataset[0]
    elif dataset_name == 'pubmed':
        transforms = T.NormalizeFeatures()
        dataset = Planetoid(root = root, name = dataset_name,transform=transforms)
        data = dataset[0]
    elif dataset_name == 'cora':
        dataset = CitationFull(root = root, name = dataset_name)
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_coauthor_amazon_splits(data, dataset.num_classes, config, lcc_mask=None)
    elif dataset_name == 'computers':
        dataset = Amazon(name = dataset_name, root = root)
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_computers_splits(data, dataset.num_classes, config)
    else:
        print('wrong dataset name')
    
    return data, dataset


def index_to_mask(index, size):
    # https://github.com/eraseai/erase/blob/master/scripts/utils.py
    """Convert index to mask."""
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_computers_splits(data, num_classes, config):
    # https://github.com/eraseai/erase/blob/master/scripts/utils.py
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing
    if not config['load_index']:
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:30] for i in indices], dim=0)
        val_index = torch.cat([i[30:50] for i in indices], dim=0)

        rest_index = torch.cat([i[50:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        now = dt.datetime.now()
        name = '../out_index/dt{}{}_{}_id{}_{}_{}_{}_noise_{}{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_tk{}_cttau{}_neigh{}{}'.format(now.month,now.day,config['dataset_name'],config['batch_id'],config['train_type'],config['algo_type'],config['module'],config['noise_type'],config['noise_rate'],config['num_layers'],config['hidden_size'],config['learning_rate'],config['max_epochs'],config['batch_size'],config['dropout'],config['ct_tk'],config['ct_tau'],config['nbr_neighbors'][0],config['nbr_neighbors'][1])
        # Save
        torch.save(train_index, name + '_train' + str(train_index.shape[0]) + '.pt')
        torch.save(val_index, name + '_val' + str(val_index.shape[0]) + '.pt')
        torch.save(rest_index, name + '_rest' + str(rest_index.shape[0]) + '.pt')
    else:
        # Load
        print('load index')
        train_index = torch.load('../out_index/dt819_computers_id1_both_coteaching_sage_noise_rand_pair0.3_lay2_hid512_lr0.001_epo50_bs512_drop0.5_tk15_cttau0.1_neigh105_train300.pt')
        val_index = torch.load('../out_index/dt819_computers_id1_both_coteaching_sage_noise_rand_pair0.3_lay2_hid512_lr0.001_epo50_bs512_drop0.5_tk15_cttau0.1_neigh105_val200.pt')
        rest_index = torch.load('../out_index/dt819_computers_id1_both_coteaching_sage_noise_rand_pair0.3_lay2_hid512_lr0.001_epo50_bs512_drop0.5_tk15_cttau0.1_neigh105_rest13252.pt')
        
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data

def random_coauthor_amazon_splits(data, num_classes, config, lcc_mask=None):
    # https://github.com/eraseai/erase/blob/master/scripts/utils.py
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing
    if not config['load_index']:
        indices = []
        if lcc_mask is not None:
            for i in range(num_classes):
                index = (data.y[lcc_mask] == i).nonzero().view(-1)
                index = index[torch.randperm(index.size(0))]
                indices.append(index)
        else:
            for i in range(num_classes):
                index = (data.y == i).nonzero().view(-1)
                index = index[torch.randperm(index.size(0))]
                indices.append(index)

        train_index = torch.cat([i[:20] for i in indices], dim=0)
        val_index = torch.cat([i[20:50] for i in indices], dim=0)

        rest_index = torch.cat([i[50:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        now = dt.datetime.now()
        name = '../out_index/dt{}{}_{}_id{}_{}_{}_{}_noise_{}{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_tk{}_cttau{}_neigh{}{}'.format(now.month,now.day,config['dataset_name'],config['batch_id'],config['train_type'],config['algo_type'],config['module'],config['noise_type'],config['noise_rate'],config['num_layers'],config['hidden_size'],config['learning_rate'],config['max_epochs'],config['batch_size'],config['dropout'],config['ct_tk'],config['ct_tau'],config['nbr_neighbors'][0],config['nbr_neighbors'][1])
        # Save
        torch.save(train_index, name + '_train' + str(train_index.shape[0]) + '.pt')
        torch.save(val_index, name + '_val' + str(val_index.shape[0]) + '.pt')
        torch.save(rest_index, name + '_rest' + str(rest_index.shape[0]) + '.pt')
    else:
        # Load
        print('load index')
        train_index = torch.load('../out_index/dt86_cora_id18_nalgo_coteaching_sage_noise_sym0.3_lay3_hid512_lr0.001_epo50_bs256_drop0.5_tk15_cttau0.2_neigh1510_train1395.pt')
        val_index = torch.load('../out_index/dt86_cora_id18_nalgo_coteaching_sage_noise_sym0.3_lay3_hid512_lr0.001_epo50_bs256_drop0.5_tk15_cttau0.2_neigh1510_val2049.pt')
        rest_index = torch.load('../out_index/dt86_cora_id18_nalgo_coteaching_sage_noise_sym0.3_lay3_hid512_lr0.001_epo50_bs256_drop0.5_tk15_cttau0.2_neigh1510_rest16349.pt')
        #train_index = torch.load('../out_index/dt812_cora_id1_baseline_coteaching_sage_noise_sym0.1_lay2_hid512_lr0.001_epo50_bs512_drop0.5_tk15_cttau0.15_neigh105_train1395.pt')
        #val_index = torch.load('../out_index/dt812_cora_id1_baseline_coteaching_sage_noise_sym0.1_lay2_hid512_lr0.001_epo50_bs512_drop0.5_tk15_cttau0.15_neigh105_val2049.pt')
        #rest_index = torch.load('../out_index/dt812_cora_id1_baseline_coteaching_sage_noise_sym0.1_lay2_hid512_lr0.001_epo50_bs512_drop0.5_tk15_cttau0.15_neigh105_rest16349.pt')
        
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data

def ogb_products_splits(data, split_idx, num_classes, config):
    if not config['load_index']:
        nbr_train_samples = int(split_idx['train'].shape[0] / config['train_frac'])
        nbr_valid_samples = int(split_idx['valid'].shape[0] / config['tvalid_frac'])
        nbr_test_samples = int(split_idx['test'].shape[0] / config['test_frac'])
        
        new_train_idx = split_idx['train'][torch.randperm(split_idx['train'].shape[0])]
        new_train_idx = new_train_idx[:nbr_train_samples]
        new_valid_idx = split_idx['valid'][torch.randperm(split_idx['valid'].shape[0])]
        new_valid_idx = new_valid_idx[:nbr_valid_samples]
        new_test_idx = split_idx['test'][torch.randperm(split_idx['test'].shape[0])]
        new_test_idx = new_test_idx[:nbr_test_samples]

        now = dt.datetime.now()
        name = '../out_index/dt{}{}_{}_id{}_{}_{}_{}_noise_{}{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_tk{}_cttau{}_neigh{}{}'.format(now.month,now.day,config['dataset_name'],config['batch_id'],config['train_type'],config['algo_type'],config['module'],config['noise_type'],config['noise_rate'],config['num_layers'],config['hidden_size'],config['learning_rate'],config['max_epochs'],config['batch_size'],config['dropout'],config['ct_tk'],config['ct_tau'],config['nbr_neighbors'][0],config['nbr_neighbors'][1])
        # Save
        torch.save(new_train_idx, name + '_train' + str(new_train_idx.shape[0]) + '.pt')
        torch.save(new_valid_idx, name + '_valid' + str(new_valid_idx.shape[0]) + '.pt')
        torch.save(new_test_idx, name + '_test' + str(new_test_idx.shape[0]) + '.pt')
    else:
        # Load
        print('load index')
        new_train_idx = torch.load('../out_index/a.pt')
        new_valid_idx = torch.load('../out_index/a.pt')
        new_test_idx = torch.load('../out_index/dt818_ogbn-products_id1_baseline_coteaching_sage_noise_next_pair0.45_lay3_hid256_lr0.001_epo50_bs512_drop0.5_tk15_cttau1.2_neigh105_test110654.pt')
    data.new_train_idx = new_train_idx
    data.new_valid_idx = new_valid_idx
    data.new_test_idx = new_test_idx
    return data