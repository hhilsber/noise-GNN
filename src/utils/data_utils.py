import os
import sys

import pickle as pkl
import torch
import scipy.sparse as sp
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

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
    if dataset_name == 'cora':
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
        stats(x, y, tx, ty, allx, ally, adjacency,labels)
        
    elif dataset_name == 'MNIST':
        datasets = torch.load('./data/MNIST/raw/MNIST.pt', weights_only=True)
        #for key,_ in datasets:
        #    print(key)
        print(datasets[0])
    else:
        graph = nx.gnp_random_graph(config['nbr_nodes'], config['edge_prob'], seed=None, directed=False)
        edge_list = []
        for (u,v,w) in graph.edges(data=True):
            w['weight'] = random.randint(0,10)
            #w['weight'] = random.uniform(0,1)
            edge_list.append([u,v])
            edge_list.append([v,u])
        edge_list = np.array(edge_list).transpose()
        
        features = np.zeros((config['nbr_nodes'],config['nbr_features']))
        for i in range(config['nbr_features']):
            attribute = 'value{}'.format(i)
            for node,_ in graph.nodes(data=True):
                features[node,i] = random.uniform(0,1)
                graph.nodes[node][attribute] = features[node,i]

        adjacency = nx.attr_matrix(graph, edge_attr="weight")[0]
        labels = np.random.randint(low=0, high=config['nbr_classes'], size=(config['nbr_nodes'], 1))

        split = int(config['nbr_nodes']*0.2)
        idx_train = torch.LongTensor(range(split))
        idx_val = torch.LongTensor(range(split, split*2))
        idx_test = torch.LongTensor(range(split*2, config['nbr_nodes']))

        adjacency = torch.from_numpy(adjacency)
        raw_features = torch.from_numpy(features)
        labels = torch.from_numpy(labels)
        edge_list = torch.from_numpy(edge_list).long()

        # Draw
        draw_graph(graph)

    data = {'adjacency': adjacency.to(device) if device else adjacency,
            'features': raw_features.to(device) if device else raw_features,
            'labels': labels.to(device) if device else labels,
            'edge_list': edge_list.to(device) if device else edge_list,
            'idx_train': idx_train.to(device) if device else idx_train,
            'idx_val': idx_val.to(device) if device else idx_val,
            'idx_test': idx_test.to(device) if device else idx_test}
        
    return data

def draw_graph(G):
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 5]
    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")
    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def nx_stats(graph):
    print("  nodes: {}".format(len(graph.nodes)))
    print("  edges: {}".format(len(graph.edges)))
    print(graph.adj)
    #adjacency = nx.adjacency_matrix(nx.from_dict_of_lists(graph.adj))
    adjacency = nx.attr_matrix(graph, edge_attr="weight")[0]
    print(adjacency.shape)
    #print(adjacency.toarray())
    print("  is weighted: {}".format(nx.is_weighted(graph)))
    print(type(nx.get_node_attributes(graph, "value1")))

    print(np.array(nx.attr_matrix(graph, node_attr="value1")[1]).reshape(-1,1))
    print(nx.attr_matrix(graph, edge_attr="weight")[0])

def stats(x, y, tx, ty, allx, ally, adjacency,labels):
    print("  x: {}".format(x.shape))
    print("  y: {}".format(y.shape))
    print("  tx: {}".format(tx.shape))
    print("  ty: {}".format(tx.shape))
    print("  allx: {}".format(allx.shape))
    print("  ally: {}".format(ally.shape))
    print("  adjacency: {}".format(adjacency.shape))
    """
    print("  ally type: {}".format(type(ally)))
    print("  labels unique: {}".format(np.unique(labels, return_counts=True)))
    print("  ally val: {}".format(ally[:3,:]))
    ttt = np.array([[0,0,0,0,0],[0,0,0,0,1]])
    print("  ttt argmax: {}".format(np.argmax(ttt, axis=1)))

    non_zero_rows = np.count_nonzero((ally != 0).sum(1))   # gives 2
    zero_rows = len(ally) - non_zero_rows
    print("  zero rows: {}".format(zero_rows))"""
    #print("  adjacency: {}".format(adjacency[1,:200]))

def stats_old(labels):

    #for key in datasets:
        #   print(key)
    #print("  adjacency shape: {}\n  features shape: {}\n  num labels: {} \n  idx_train: {}\n  idx_val: {}\n  idx_test: {} \n ".format(datasets['adjacency'].shape,datasets['features'].shape,datasets['labels'].max().item() + 1,datasets['idx_train'].shape,datasets['idx_val'].shape,datasets['idx_test'].shape))
    print("  labels: {}\n".format(labels.max().item() + 1))
    #print("  labels: {}\n".format(labels))