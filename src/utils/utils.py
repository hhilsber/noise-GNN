import logging
import matplotlib
import os
import torch
import numpy as np
import scipy.sparse as sp

def to_scipy_sparse_matrix(edge_index, num_nodes):
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.ones(len(row))
    shape = (num_nodes, num_nodes)
    return sp.coo_matrix((data, (row, col)), shape=shape)

def to_scipy_sparse_matrix_rcd(row, col, data, num_nodes):
    shape = (num_nodes, num_nodes)
    return sp.coo_matrix((data, (row, col)), shape=shape)

def initialize_logger(config, file_name):
    # Clear handlers if they were created in other runs
    logging.getLogger().handlers = []

    # Set up the logger
    log_name = '../out_logs/' + config['algo_type'] + config['what'] + '/' + file_name + '.log'
    
    # Clear log file if it exists
    if os.path.exists(log_name):
        os.remove(log_name)

    logging.basicConfig(filename=log_name, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Suppress debug-level messages from Matplotlib
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)
    
    logger = logging.getLogger()
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        logger.debug("{} -->   {}".format(keystr, val))
    
    return logger


"""
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
    """