import numpy as np
import torch_geometric.utils as tg

def sample_neighbor(dataset, k_hop=2):
    nnbr = 0 # node number
    nlbl = dataset['labels'][nnbr] # node label
    subset, edge_index, mapping, edge_mask = tg.k_hop_subgraph(node_idx=nnbr, num_hops=k_hop, edge_index=dataset['edge_list'], relabel_nodes=True)
    same_class = np.where(dataset['labels']==nlbl)[0]
    print(subset)
    print(same_class)
    print(np.intersect1d(subset, same_class))