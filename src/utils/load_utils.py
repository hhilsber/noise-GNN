import os
import sys

import pickle as pkl
import torch
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, CitationFull
import torch_geometric.transforms as T

def random_coauthor_amazon_splits(data, num_classes, lcc_mask=None):
    # https://github.com/eraseai/erase/blob/master/scripts/utils.py
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing
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

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

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
    elif dataset_name == 'citationfull':
        dataset = CitationFull(path, name = "cora")
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_coauthor_amazon_splits(
            data, dataset.num_classes, lcc_mask=None)
    else:
        print('wrong dataset name')
    
    return data, dataset