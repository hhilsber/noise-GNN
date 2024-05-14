import torch
import numpy as np
import random
import torch.nn.functional as F

def flip_label(labels, nbr_classes, noise_type='sym', prob=0.3):
    """
    labels
    prob: probability to flip to another class
    """
    if noise_type == 'sym':
        noise_mat = np.diag(np.array([1-prob] * nbr_classes), k=0) + (np.ones((nbr_classes,nbr_classes)) - np.diag(np.ones(nbr_classes), k=0)) * (prob/(nbr_classes-1))
    elif noise_type == 'next_pair':
        noise_mat = np.diag(np.array([1-prob] * nbr_classes), k=0) + np.diag(np.array([prob] * (nbr_classes-1)), k=1) + np.diag(np.array([prob]), k=-(nbr_classes-1))
    elif noise_type == 'rand_pair':
        perm1 = np.random.permutation(nbr_classes)
        perm2 = np.random.permutation(nbr_classes)
        numbers = np.arange(nbr_classes)
        
        row1 = numbers[perm1]
        row2 = numbers[perm2]
        random_perm = np.vstack((row1, row2))

        noise_mat = np.diag(np.array([1-prob] * nbr_classes), k=0)
        mat = np.zeros((nbr_classes,nbr_classes))
        for i in range(nbr_classes):
            mat[random_perm[0,i],random_perm[1,i]] = prob
        noise_mat += mat
    elif noise_type == 'aim_pair':
        existing_array = np.array([[0, 1, 2], [3, 4, 5]])
        m = existing_array.shape[1]
        perm1 = np.random.permutation(nbr_classes - m)
        perm2 = np.random.permutation(nbr_classes - m)
        
        numbers = np.arange(nbr_classes)
        numbers1 = np.setdiff1d(numbers, existing_array[0,:])
        numbers2 = np.setdiff1d(numbers, existing_array[1,:])
        
        if (len(numbers1) == 0):
            return np.array([])

        row1 = numbers1[perm1]
        row2 = numbers2[perm2]
        random_perm = np.concatenate((np.vstack((row1, row2)), existing_array), axis=1)

        noise_mat = np.diag(np.array([1-prob] * nbr_classes), k=0)
        mat = np.zeros((nbr_classes,nbr_classes))
        for i in range(nbr_classes):
            mat[random_perm[0,i],random_perm[1,i]] = prob
        noise_mat += mat
    else:
        print('wrong noise type')
        
    noisy_labels = np.copy(labels.numpy())
    for i in range(labels.shape[0]):
        lbl = labels[i].item()
        flipped = np.random.multinomial(1, noise_mat[lbl,:], 1)[0]
        noisy_labels[i] = np.where(flipped == 1)[0]
    noisy_labels = torch.from_numpy(noisy_labels).squeeze()

    return noisy_labels.squeeze(), noise_mat

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