import os
import sys

import pickle as pkl
import torch
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset

from .data_utils import edges_to_adjacency

def load_network(config):
    """
    txt
    """

    device = config['device']
    data_dir = config['data_dir']
    dataset_name = config['dataset_name']
    
    if dataset_name == 'cora':
        dataset = pkl.load(open(f'{data_dir}/{dataset_name}/dataset.pkl', 'rb'))
        
        features = dataset.x
        labels = dataset.y
        edge_index = dataset.edge_index
        train_mask = dataset.train_mask
        val_mask = dataset.val_mask
        test_mask = dataset.test_mask
        
        adjacency = edges_to_adjacency(edge_index, features.shape[0])

        data = {'adjacency': adjacency.to(device) if device else adjacency,
            'features': features.to(device) if device else features,
            'labels': labels.to(device) if device else labels,
            'edge_index': edge_index.to(device) if device else edge_index,
            'train_mask': train_mask.to(device) if device else train_mask,
            'val_mask': val_mask.to(device) if device else val_mask,
            'test_mask': test_mask.to(device) if device else test_mask}
    if dataset_name == 'ogbn-arxiv':
        root = 'C:/Users/hendr/Documents/data'
        dataset = PygNodePropPredDataset('ogbn-arxiv', root)
        #dataset = pkl.load(open(f'{data_dir}/{dataset_name}/dataset.pkl', 'rb'))
        #print("total nodes: {} --- split train/val/test {}/{}/{}".format(dataset.x.shape[0],dataset.train_idx.shape[0],dataset.valid_idx.shape[0],dataset.test_idx.shape[0]))
    
    if dataset_name == 'ogbn-mag':
        root = config['data_dir']
        dataset = PygNodePropPredDataset('ogbn-mag', root)
    if dataset_name == 'ogbn-products':
        root = config['data_dir']
        dataset = PygNodePropPredDataset('ogbn-products', root)
    return dataset