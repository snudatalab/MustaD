import torch
import copy
from tqdm import tqdm
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from typing import Optional, List, Union
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from torch.utils.checkpoint import checkpoint
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch.nn import BatchNorm1d, LayerNorm, InstanceNorm1d
from torch_sparse import SparseTensor
from torch_scatter import scatter, scatter_softmax
from torch_geometric.nn.conv import MessagePassing
from utils import *

class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        """
        Construct teacher deeperGCN model
        
        :param hidden_channels: teacher's hidden dimension
        :param num_layers: number of layers in the teacher
        """
        super(DeeperGCN, self).__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1000, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        """
        Forward x to the task prediction
        
        :param x: node features
        :param edge_index: adjacency matrix index
        :param edge_attr: weight for each adjacency matrix index
        :return: task prediction, hidden representation set
        """
        xs = list()
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            xs.append(x)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)
        xs.append(x)

        return self.lin(x), xs

class student(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        """
        Construct student deeperGCN model
        
        :param hidden_channels: student's hidden dimension
        :param num_layers: number of layers in the student
        """
        super(student, self).__init__()

        self.num_layers = num_layers
        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers+1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1000, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        """
        Forward x to the task prediction
        
        :param x: node features
        :param edge_index: adjacency matrix index
        :param edge_attr: weight for each adjacency matrix index
        :return: task prediction, hidden representation set
        """
        xs = list()
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for i in range(nl-1):
            x = self.layers[1](x, edge_index, edge_attr)
            xs.append(x)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)
        xs.append(x)

        return self.lin(x), xs
