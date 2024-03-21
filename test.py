import torch

ttt = torch.rand((3,3))
print(ttt)
adj = torch.tensor([[0,1,0],[1,1,1],[0,0,1]])
print(adj)
q1= 0.25
q3= 0.75
quant1 = torch.quantile(ttt, q1)
quant3 = torch.quantile(ttt, q3)
print(quant1,quant3)
new_adj = torch.where(ttt < quant1, 0, adj)
new_adj = torch.where(ttt > quant3, 1, new_adj)
print(new_adj)