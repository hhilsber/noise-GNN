import os
import numpy as np
import pickle as pkl
import torch
import scipy.sparse as sp

def load_network(config):
    """
    txt
    """

    data = {}
    device = config['device']

    #adjacency = pkl.load(open(f'{config['data_dir']}{config['dataset_name']}_adj.pkl', 'rb'))
    #adjacency = pkl.load(open(os.path.join(config['data_dir'], config['dataset_name'], "/{}_adj.pkl".format()), 'rb'))
    #adjacency = pkl.load(open("{}{}_adj.pkl".format(config['data_dir'], config['dataset_name']), 'rb'))
    adjacency = pkl.load(open("./data/cora/cora_adj.pkl", 'rb'))
    data = {'adjacency': adjacency.to(device) if device else adjacency}

    return data