import torch
import torch.nn as nn
import torch.nn.functional as F

class SIMA(nn.Module):
    def __init__(self, nbr_nodes, nbr_features, dropout=0.5):
        super(SIMA, self).__init__()
        self.weight_i = nn.Parameter(torch.randn(nbr_features, nbr_nodes))
        self.weight_j = nn.Parameter(torch.randn(nbr_features, nbr_nodes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features):
        # Compute the attention scores for each pair of nodes
        attention_i = self.dropout(torch.relu(torch.matmul(node_features, self.weight_i)))
        attention_j = self.dropout(torch.relu(torch.matmul(node_features, self.weight_j)))
        attention_j = attention_j.transpose(0, 1)
        
        # Compute the similarity matrix
        similarity_matrix = torch.matmul(attention_i, attention_j)
        return similarity_matrix

class AttentionLayer(nn.Module):
    """
    graph attention
    """

    def __init__(self, in_size, out_size, dropout, out_act, alpha=0.2, concat=True):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_size = in_size
        self.out_size = out_size
        self.alpha = alpha
        self.concat = concat # multi head

        self.W = nn.Parameter(torch.zeros(size=(in_size, out_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a1 = nn.Parameter(torch.zeros(size=(out_size, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(out_size, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        if out_act == 'relu':
            self.out_act = nn.ReLU()
        else:
            self.out_act = nn.ELU()

    def forward(self, inp, adj):
        h = torch.mm(inp, self.W)
        print('    att nan i1: {}'.format(torch.count_nonzero(torch.isnan(self.W))))
        print('    att nan i2: {}'.format(torch.count_nonzero(torch.isnan(h))))
        N = h.size()[0]

        a_input1 = torch.matmul(h, self.a1)
        a_input2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(a_input1 + a_input2.transpose(-1, -2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, nnode, nfeat, nclass, nhid, dropout, out_act, alpha, nheads):
        """Graph Attention network"""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [AttentionLayer(nfeat, nhid, dropout=dropout, out_act=None, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        #self.out_att = AttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.out_att = AttentionLayer(nhid * nheads, nnode, dropout=dropout, out_act=out_act, alpha=alpha, concat=False)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att.out_act(self.out_att(x, adj))
        return x