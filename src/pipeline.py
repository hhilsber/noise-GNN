import numpy as np
import torch

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
            #self.device = torch.device('cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
            #cudnn.benchmark = True
            self.device = torch.device("cuda")
        else:
            self.device = torch.device('cpu')
        print("Cuda?  is_available: {} --- version: {} --- device: {}".format(is_cuda,torch.version.cuda, self.device))
        config['device'] = self.device


        datasets = load_network(config)