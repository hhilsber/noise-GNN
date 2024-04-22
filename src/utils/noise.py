import torch
import numpy as np
import random


def flip_label(labels, nbr_classes, noise_type='rand', prob=0.3):
    """
    labels
    prob: probability to flip to another class
    """
    if noise_type == 'rand':
        new_y = np.copy(F.one_hot(labels, nbr_classes).squeeze().numpy())
        for i in range(new_y.shape[0]):
            if np.random.rand() <= prob:
                class_prob = (np.ones((new_y.shape[1])) - new_y[i,:]) * (1/(new_y.shape[1]-1))
                flipped = np.random.multinomial(1, class_prob, 1)[0]
                new_y[i] = flipped
        noisy_labels = torch.from_numpy(new_y).argmax(dim=-1)
    elif noise_type == 'pair':
        P = np.diag(np.array([1-prob] * nbr_classes), k=0) + np.diag(np.array([prob] * (nbr_classes-1)), k=1) + np.diag(np.array([prob]), k=-(nbr_classes-1))

        noisy_labels = np.copy(labels.numpy())
        for i in range(labels.shape[0]):
            lbl = labels[i].item()
            flipped = np.random.multinomial(1, P[lbl,:], 1)[0]
            noisy_labels[i] = np.where(flipped == 1)[0]
    return noisy_labels

def add_edge_noise(adjacency, prob=0.4):
    """
    adjacency: adjacency matrix
    prob: probability to delete and add edges
    """
    
    new_adj = np.copy(adjacency.numpy())
    init_nbr_edges = np.sum(new_adj)
    

    if (new_adj == np.transpose(new_adj)).all():
        # Triangle matrix
        trgl = np.triu(new_adj, k=1)
        row, col = trgl.nonzero()
        nbr_edges = row.shape[0]
        modify_nbr = int(nbr_edges * prob)
        
        # Delete
        delete_idx = random.sample(range(nbr_edges), k=modify_nbr)
        delete_edges = np.transpose(np.array([row[delete_idx],col[delete_idx]]))
        trgl_delete = np.triu(new_adj, k=1)
        trgl_delete[tuple(delete_edges.T)] = 0.

        # Add
        potential_add_edges = np.triu(np.ones_like(new_adj)-new_adj, k=1)
        row, col = potential_add_edges.nonzero()
        add_idx = random.sample(range(nbr_edges), k=modify_nbr)
        add_edges = np.transpose(np.array([row[add_idx],col[add_idx]]))
        trgl_add = np.zeros_like(new_adj)
        trgl_add[tuple(add_edges.T)] = 1.

        new_adj = trgl_delete + np.transpose(trgl_delete) + trgl_add + np.transpose(trgl_add)
        return torch.from_numpy(new_adj)

def add_feature_noise(features, prob, mean=0, std=0.1):
    """
    Adds Gaussian noise to a feature matrix with probabilistic masking.

    Args:
    - features (numpy.ndarray): Feature matrix.
    - p (float): Probability of changing each element.
    - mean (float): Mean of the Gaussian distribution.
    - std (float): Standard deviation of the Gaussian distribution.

    Returns:
    - noisy_features (numpy.ndarray): Feature matrix with Gaussian noise added.
    """
    r,_ = torch.max(features, dim=1)
    print(r)
    noise = np.random.normal(mean, std, size=features.shape)
    mask = np.random.choice([0, 1], size=features.shape, p=[1-prob, prob])
    noisy_features = features + mask * noise
    return noisy_features