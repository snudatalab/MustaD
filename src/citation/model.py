import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        """
        Construct GCNII layer

        :param in_features: input dimension
        :param out_features: output dimension
        :param residual: ratio of residual connection
        :param variant: variant version introduced in GCNII
        """
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters
        
        :return: none
        """
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        """
        Compute a GCNII layer
        
        :param input: input feature
        :param adj: adjacency matrix
        :param lamda: ratio of lamda
        :param alpha: alpha
        :param l: l^{th} layer in an overall model
        """
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        """
        Constructor of GCNII teacher model

        :param nfeat: input dimension
        :param nlayers: number of layers
        :param nhidden: hidden feature dimension
        :param nclass: number of output class
        :param dropout: ratio of dropout
        :param lamda: ratio of lamda
        :param alpha: alpha
        :param variant: variant version introduced in GCNII
        """
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        """
        Forward x into class

        :param x: input node features
        :param adj: adjacency matrix
        :return: task prediction, last hidden embedding
        """
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        hidden_emb = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](hidden_emb)
        return layer_inner, hidden_emb

class GCNII_student(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, thidden, nclass, dropout, lamda, alpha, variant):
        """
        Constructor of GCNII student model

        :param nfeat: input dimension
        :param nlayers: number of layers
        :param nhidden: student's hidden feature dimension
        :param thidden: teacher's hidden feature dimension
        :param nclass: number of output class
        :param dropout: ratio of dropout
        :param dropout: ratio of lamda
        :param alpha: alpha
        :param variant: variant version introduced in GCNII
        """
        super(GCNII_student, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.tlayers = nlayers
        self.nhidden = nhidden
        self.thidden = thidden
        if self.nhidden != self.thidden:
            self.match_dim = nn.Linear(nhidden, thidden)

    def forward(self, x, adj):
        """
        Forward x into class

        :param x: input node features
        :param adj: adjacency matrix
        :return: task prediction, last hidden embedding
        """
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i in range(self.tlayers):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(self.convs[0](layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        hidden_emb = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](hidden_emb)
        if self.nhidden != self.thidden:
            hidden_emb = self.match_dim(hidden_emb)

        return layer_inner, hidden_emb


if __name__ == '__main__':
    pass






