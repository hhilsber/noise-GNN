import pickle
import argparse
import yaml
import random

import numpy as np
import torch
import torch.nn as nn

from src.pipeline import PipelineCO
from src.pipeline_ctp import PipelineCTP
from src.pipeline_contrast import PipelineCT
from src.pipeline_test import PipelineTE
from src.pipeline_h import PipelineH
from src.pipeline_s import PipelineS
from src.pipeline_test_s import PipelineTES

##################################### Main #####################################
def main(config):
    for bs in [32,64,128,256]:
        for dp in [0.1,0.25,0.5]:
            for hid in [128,256,1024]:
                config['batch_size'] = bs
                config['dropout'] = dp
                config['hidden_size'] = hid
                ##
                if config['algo_type'] in ['codi', 'coteaching']:
                    if config['what'] == '_test2':
                        if config['dataset_name'] in ['ogbn-arxiv']:
                            model = PipelineTE(config)
                        else:
                            print('pipeline TES')
                            model = PipelineTES(config)
                    else:
                        if config['dataset_name'] in ['ogbn-arxiv']:
                            model = PipelineCO(config)
                        else:
                            model = PipelineS(config)
                elif config['algo_type'] == 'ctp':
                    model = PipelineCTP(config)
                elif config['algo_type'] == 'contrastive':
                    model = PipelineCT(config)
                else:
                    print('wrong algo type')
                model.loop()


##################################### Fcts #####################################
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='configuration file path')
    args = vars(parser.parse_args())
    return args

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.Loader)
    return config

def show_config(config):
    print("----------------- CONFIG -----------------")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("----------------- CONFIG -----------------")

##################################### Cmdl #####################################
if __name__ == '__main__':


    ppl_args = get_arguments()
    config = get_config(ppl_args['config'])
    #show_config(config)

    # Set device
    is_cuda = torch.cuda.is_available()
    if config['cuda'] and is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Cuda?  is_available: {} --- version: {} --- device: {}'.format(is_cuda,torch.version.cuda, device))
    
    config['device'] = device

    main(config)