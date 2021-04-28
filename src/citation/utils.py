import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx
import json
from networkx.readwrite import json_graph
import pdb
sys.setrecursionlimit(99999)

def count_params(model):
    """
    Count the number of parameters.

    :param model: model
    :return: number of parameters of the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def kernel(t_x, s_x, kernel):
    """
    Apply kernel function and return the distance

    :param t_x: feature vector
    :param s_x: feature vector
    :return: kernel based distance between two input vectors
    """
    dot = torch.bmm(t_x.view(t_x.size()[0], 1, t_x.size()[1]), s_x.view(s_x.size()[0], s_x.size()[1], 1)).squeeze()
    if kernel=='kl':
        kl_loss_op = torch.nn.KLDivLoss(reduction='none')
        t_x = F.softmax(t_x, dim=1)
        s_x = F.log_softmax(s_x, dim=1)
        return torch.mean(torch.sum(kl_loss_op(s_x, t_x), dim=1))
    elif kernel=='lin':
        return torch.mean(dot)
    elif kernel=='poly':
        return torch.mean(dot*dot)
    elif kernel=='dist':
        return torch.mean(torch.norm(t_x-s_x, dim=1))
    elif kernel=='RBF':
        return torch.mean(torch.exp(-1/2 * torch.norm(t_x-s_x, dim=1)))

def accuracy(output, labels):
    """
    Compute accuracy

    :param output: prediction
    :param labels: labels
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def normalize(mx):
    """
    Row-normalize sparse matrix

    :param mx: matrix
    :return: Row-normalize sparse matrix
    """
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.

    :param sparse_mx: scipy sparse matrix
    :return: torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """
    Parse index file.

    :param filename: file name
    :return: parsed index
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# adapted from tkipf/gcn
def load_citation(dataset_str="cora"):
    """
    Load Citation Networks Datasets.
    :param dataset_str: dataset
    :return: adjacency, node features, node labels, train index, validation index, test index
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/citation/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/citation/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    features = normalize(features)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    adj = sys_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels, idx_train, idx_val, idx_test

# adapted from PetarV/GAT
def run_dfs(adj, msk, u, ind, nb_nodes):
    if msk[u] == -1:
        msk[u] = ind
        for v in adj[u,:].nonzero()[1]:
            run_dfs(adj, msk, v, ind, nb_nodes)

def dfs_split(adj):
    # Assume adj is of shape [nb_nodes, nb_nodes]
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)

    graph_id = 0

    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1

    return ret

def test(adj, mapping):
    nb_nodes = adj.shape[0]
    for i in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] != mapping[j]:
                 return False
    return True

def find_split(adj, mapping, ds_label):
    nb_nodes = adj.shape[0]
    dict_splits={}
    for i in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i]==0 or mapping[j]==0:
                dict_splits[0]=None
            elif mapping[i] == mapping[j]:
                if ds_label[i]['val'] == ds_label[j]['val'] and ds_label[i]['test'] == ds_label[j]['test']:

                    if mapping[i] not in dict_splits.keys():
                        if ds_label[i]['val']:
                            dict_splits[mapping[i]] = 'val'

                        elif ds_label[i]['test']:
                            dict_splits[mapping[i]]='test'

                        else:
                            dict_splits[mapping[i]] = 'train'

                    else:
                        if ds_label[i]['test']:
                            ind_label='test'
                        elif ds_label[i]['val']:
                            ind_label='val'
                        else:
                            ind_label='train'
                        if dict_splits[mapping[i]]!= ind_label:
                            print ('inconsistent labels within a graph exiting!!!')
                            return None
                else:
                    print ('label of both nodes different, exiting!!')
                    return None
    return dict_splits

