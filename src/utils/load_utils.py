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
        """
        print(dataset_name)
        
        dataset = PygNodePropPredDataset(name = dataset_name)
        #graph = dataset[0]

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        dataset.train_idx = train_idx
        dataset.valid_idx = valid_idx
        dataset.test_idx = test_idx

        #adjacency = torch.tensor([0.])
        #data = {'adjacency': adjacency.to(device) if device else adjacency}"""
        dataset = pkl.load(open(f'{data_dir}/{dataset_name}/dataset.pkl', 'rb'))
        print(dataset.valid_idx.shape)
    return dataset