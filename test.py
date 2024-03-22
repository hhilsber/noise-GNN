import torch
from src.utils.losses import *

n=3
A = torch.randint(0, 5, (n,n)).float()
X = torch.rand((n,n))
print(A)
D = compute_degree_matrix(A)
L = compute_laplacian_matrix(D, A)
print(compute_smoothness(X, L, n))
print(compute_connectivity(A, n))
print(compute_sparsity(A, n))

