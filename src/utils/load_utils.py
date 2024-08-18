import os
import sys

import pickle as pkl
import torch
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, CitationFull
import torch_geometric.transforms as T
import datetime as dt

def index_to_mask(index, size):
    # https://github.com/eraseai/erase/blob/master/scripts/utils.py
    """Convert index to mask."""
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

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
        #train_index = torch.load('../out_index/dt86_cora_id18_nalgo_coteaching_sage_noise_sym0.3_lay3_hid512_lr0.001_epo50_bs256_drop0.5_tk15_cttau0.2_neigh1510_train1395.pt')
        #val_index = torch.load('../out_index/dt86_cora_id18_nalgo_coteaching_sage_noise_sym0.3_lay3_hid512_lr0.001_epo50_bs256_drop0.5_tk15_cttau0.2_neigh1510_val2049.pt')
        #rest_index = torch.load('../out_index/dt86_cora_id18_nalgo_coteaching_sage_noise_sym0.3_lay3_hid512_lr0.001_epo50_bs256_drop0.5_tk15_cttau0.2_neigh1510_rest16349.pt')
        train_index = torch.load('../out_index/dt812_cora_id1_baseline_coteaching_sage_noise_sym0.1_lay2_hid512_lr0.001_epo50_bs512_drop0.5_tk15_cttau0.15_neigh105_train1395.pt')
        val_index = torch.load('../out_index/dt812_cora_id1_baseline_coteaching_sage_noise_sym0.1_lay2_hid512_lr0.001_epo50_bs512_drop0.5_tk15_cttau0.15_neigh105_val2049.pt')
        rest_index = torch.load('../out_index/dt812_cora_id1_baseline_coteaching_sage_noise_sym0.1_lay2_hid512_lr0.001_epo50_bs512_drop0.5_tk15_cttau0.15_neigh105_rest16349.pt')
        
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data

def ogb_products_splits(data, split_idx, num_classes, config):
    if not config['load_index']:
        nbr_test_samples = int(split_idx['test'].shape[0] / config['test_frac'])
        new_test_split = split_idx['test'][torch.randperm(split_idx['test'].shape[0])]
        new_test_split = new_test_split[:nbr_test_samples]

        now = dt.datetime.now()
        name = '../out_index/dt{}{}_{}_id{}_{}_{}_{}_noise_{}{}_lay{}_hid{}_lr{}_epo{}_bs{}_drop{}_tk{}_cttau{}_neigh{}{}'.format(now.month,now.day,config['dataset_name'],config['batch_id'],config['train_type'],config['algo_type'],config['module'],config['noise_type'],config['noise_rate'],config['num_layers'],config['hidden_size'],config['learning_rate'],config['max_epochs'],config['batch_size'],config['dropout'],config['ct_tk'],config['ct_tau'],config['nbr_neighbors'][0],config['nbr_neighbors'][1])
        # Save
        torch.save(new_test_split, name + '_test' + str(new_test_split.shape[0]) + '.pt')
    else:
        # Load
        print('load index')
        new_test_split = torch.load('../out_index/a.pt')
    data.new_test_split = new_test_split
    return data

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
    else:
        print('wrong dataset name')
    
    return data, dataset