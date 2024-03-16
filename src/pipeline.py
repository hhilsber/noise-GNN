import numpy as np
import torch
import torch_geometric.utils as tg

from .utils.data_utils import load_network

class Pipeline(object):
    """
    Processing pipeline
    """
    def __init__(self, config):
        # Set metrics:
    
        #Set device
        is_cuda = torch.cuda.is_available()
        if not config['no_cuda'] and is_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print('Cuda?  is_available: {} --- version: {} --- device: {}'.format(is_cuda,torch.version.cuda, self.device))
        config['device'] = self.device


        datasets = load_network(config)
        print(datasets['edge_list'])
        subset, edge_index, mapping, edge_mask = tg.k_hop_subgraph(node_idx=0, num_hops=1, edge_index=datasets['edge_list'], relabel_nodes=True)
        print(subset)