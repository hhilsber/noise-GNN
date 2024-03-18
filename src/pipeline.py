import torch
import numpy as np
import torch_geometric.utils as tg

from .utils.load_utils import load_network
from .utils.data_utils import sample_neighbor

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
        
        
        dataset = load_network(config)
        neighbors = sample_neighbor(dataset)