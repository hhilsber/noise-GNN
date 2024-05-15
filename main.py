import pickle
import argparse
import yaml
import random

import numpy as np
import torch
import torch.nn as nn

from src.pipeline_prod import PipelineP
from src.pipeline_arx import PipelineA
from src.pipeline_sh import PipelineSH

##################################### Main #####################################
def main(config):
    if (config['dataset_name'] == 'ogbn-products') and config['algo_type'] in ['ct', 'codi']:
        model = PipelineP(config)
    elif (config['dataset_name'] == 'ogbn-products') and config['algo_type'] in ['cn_soft', 'cn_hard']:
        model = PipelineSH(config)
    else:
        model = PipelineA(config)
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