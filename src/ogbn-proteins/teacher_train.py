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
from model import *
import argparse

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--lbd_pred', type=float, default=0.1, help='lambda for prediction loss')
parser.add_argument('--lbd_embd', type=float, default=0.01, help='lambda for embedding loss')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions')
parser.add_argument('--layer', type=int, default=28, help='Number of teacher layers')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--train_bn', type=int, default=40, help='train batch number')
parser.add_argument('--test_bn', type=int, default=5, help='Test batch number')
args = parser.parse_args()

# Load data
dataset = PygNodePropPredDataset('ogbn-proteins', root='data')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.node_species = None
data.y = data.y.to(torch.float)
train_loader = RandomNodeSampler(data, num_parts=40, shuffle=True, num_workers=5)
test_loader = RandomNodeSampler(data, num_parts=5, num_workers=5)

# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')

# Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask

# Define models
device = torch.device(f'cuda:{args.dev}' if torch.cuda.is_available() else 'cpu')
model = DeeperGCN(hidden_channels=args.hidden, num_layers=args.layer).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
evaluator = Evaluator('ogbn-proteins')
PATH = "./teacher/teacher_ogbn" + str(args.layer) + ".pth"


def train(epoch):
    """
    Start training student.
    
    :param epoch: current train epoch
    :make sure model, optimizers, node features, adjacency, train index is defined aforehead
    :return: train loss
    """
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test():
    """
    test the model on training dataset, validation dataset, test dataset.
    
    :make sure model is defined aforehead.
    :return: ROC_AUC on training data, ROC_AUC on validation data, ROC_AUC on test data
    """
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


# Start training
max_acc = [0, 0, 0]
for epoch in range(1, 1001):
    loss = train(epoch)
    train_rocauc, valid_rocauc, test_rocauc = test()
    if(valid_rocauc>max_acc[1]):
        max_acc[0] = train_rocauc
        max_acc[1] = valid_rocauc
        max_acc[2] = test_rocauc
        torch.save(model.state_dict(), PATH)
    print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
          f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')

print(f'#Parameters: {count_parameters(model):02d}, Train: {max_acc[0]:.4f}, '
    f'Val: {max_acc[1]:.4f}, Test: {max_acc[2]:.4f}')
