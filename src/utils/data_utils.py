import numpy as np
import torch_geometric.utils as tg

def sample_neighbor(dataset, k_hop=2):
    nnbr = 0 # node number
    nlbl = dataset['labels'][nnbr] # node label
    subset, edge_index, mapping, edge_mask = tg.k_hop_subgraph(node_idx=nnbr, num_hops=k_hop, edge_index=dataset['edge_list'], relabel_nodes=True)
    
    same_class = np.where(dataset['labels']==nlbl)[0]
    print("  subset: {}".format(subset))
    print("  same_class: {}".format(same_class))
    inter = np.intersect1d(subset.tolist(), same_class)
    print("  intersect: {}".format(inter))
    res = [item for item in subset.tolist() if item not in inter]
    print("  diff inter: {}".format(res))