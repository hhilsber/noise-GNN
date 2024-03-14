import os
import sys
import numpy as np
import pickle as pkl
import torch
import scipy.sparse as sp
import networkx as nx

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_network(config):
    """
    txt
    """

    #data = {}
    device = config['device']
    data_dir = config['data_dir']
    dataset_name = config['dataset_name']
    #adjacency = pkl.load(open(f'{config['data_dir']}{config['dataset_name']}_adj.pkl', 'rb'))
    #adjacency = pkl.load(open(os.path.join(config['data_dir'], config['dataset_name'], "/{}_adj.pkl".format()), 'rb'))
    #adjacency = pkl.load(open("{}{}_adj.pkl".format(config['data_dir'], config['dataset_name']), 'rb'))
    #x = pkl.load(open("./data/cora/raw/ind.cora.x", 'rb'))
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_dir, 'ind.{}.{}'.format(dataset_name, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    adjacency = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    test_idx_reorder = parse_index_file(os.path.join(data_dir, 'ind.{}.test.index'.format(dataset_name)))
    test_idx_range = np.sort(test_idx_reorder)

    raw_features = sp.vstack((allx, tx)).tolil() #vertical concat
    raw_features[test_idx_reorder, :] = raw_features[test_idx_range, :]
    raw_features = torch.Tensor(raw_features.todense())
    #features = normalize_features(raw_features)
    #features = torch.Tensor(features.todense())

    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # labels = torch.LongTensor(np.where(labels)[1])
    labels = torch.LongTensor(np.argmax(labels, axis=1))

    idx_train = torch.LongTensor(range(len(y)))
    idx_val = torch.LongTensor(range(len(y), len(y) + 500))
    idx_test = torch.LongTensor(test_idx_range.tolist())
    
    adjacency = torch.Tensor(adjacency.todense())
    #return adj_norm, features, labels, idx_train, idx_val, idx_test
    #stats(x, y, tx, ty, allx, ally, adjacency,labels)
    data = {'adjacency': adjacency.to(device) if device else adjacency,
            'features': raw_features.to(device) if device else raw_features,
            'labels': labels.to(device) if device else labels,
            'idx_train': idx_train.to(device) if device else idx_train,
            'idx_val': idx_val.to(device) if device else idx_val,
            'idx_test': idx_test.to(device) if device else idx_test}

    return data


def stats(x, y, tx, ty, allx, ally, adjacency,labels):
    print("  x: {}".format(x.shape))
    print("  y: {}".format(y.shape))
    print("  tx: {}".format(tx.shape))
    print("  ty: {}".format(tx.shape))
    print("  allx: {}".format(allx.shape))
    print("  ally: {}".format(ally.shape))
    print("  adjacency: {}".format(adjacency.shape))

    print("  ally type: {}".format(type(ally)))
    print("  labels unique: {}".format(np.unique(labels, return_counts=True)))
    print("  ally val: {}".format(ally[:3,:]))
    ttt = np.array([[0,0,0,0,0],[0,0,0,0,1]])
    print("  ttt argmax: {}".format(np.argmax(ttt, axis=1)))

    non_zero_rows = np.count_nonzero((ally != 0).sum(1))   # gives 2
    zero_rows = len(ally) - non_zero_rows
    print("  zero rows: {}".format(zero_rows))

def stats_old(labels):

    #for key in datasets:
        #   print(key)
    #print("  adjacency shape: {}\n  features shape: {}\n  num labels: {} \n  idx_train: {}\n  idx_val: {}\n  idx_test: {} \n ".format(datasets['adjacency'].shape,datasets['features'].shape,datasets['labels'].max().item() + 1,datasets['idx_train'].shape,datasets['idx_val'].shape,datasets['idx_test'].shape))
    print("  labels: {}\n".format(labels.max().item() + 1))
    #print("  labels: {}\n".format(labels))