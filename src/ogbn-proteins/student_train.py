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
parser.add_argument('--layer', type=int, default=64, help='Number of teacher layers')
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
train_loader = RandomNodeSampler(data, num_parts=args.train_bn, shuffle=True, num_workers=5)
test_loader = RandomNodeSampler(data, num_parts=args.test_bn, num_workers=5)

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
t_model = DeeperGCN(hidden_channels=args.hidden, num_layers=args.layer).to(device)
s_model = student(hidden_channels=args.hidden, num_layers=2).to(device)
t_optimizer = torch.optim.Adam(t_model.parameters(), lr=0.01)
s_optimizer = torch.optim.Adam(s_model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
evaluator = Evaluator('ogbn-proteins')
t_PATH = "./src/ogbn-proteins/teacher/teacher_ogbn" + str(args.layer) + ".pth"
s_PATH = "./src/ogbn-proteins/student/student_ogbn" + str(args.layer) + ".pth"


def train_student_model(t_model, epoch):
    """
    Start training student.
    
    :param t_model: trained teacher model
    :param epoch: current train epoch
    :make sure student, optimizers, node features, adjacency, train index is defined aforehead
    :return: train loss
    """
    kl_loss_op = torch.nn.KLDivLoss(reduction='none')
    temperature = 2

    s_model.train()
    t_model.eval()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        s_optimizer.zero_grad()
        data = data.to(device)

        # loss_prediction
        s_logits, s_xs = s_model(data.x, data.edge_index, data.edge_attr)
        t_logits, t_xs = t_model(data.x, data.edge_index, data.edge_attr)
        t_logits = t_logits / temperature
        s_y = F.log_softmax(s_logits[data.train_mask], dim=1)
        t_y = F.softmax(t_logits[data.train_mask], dim=1)
        pred_loss = kl_loss_op(s_y, t_y)
        pred_loss = torch.mean(torch.sum(kl_loss, dim=1))

        # loss_BCE
        BCE_loss = criterion(s_logits[data.train_mask], data.y[data.train_mask])

        # loss_embedding
        t_x = F.softmax(t_xs[-1][data.train_mask], dim=1)
        s_x = F.log_softmax(s_xs[-1][data.train_mask], dim=1)
        embd_loss = torch.mean(torch.sum(kl_loss_op(s_x, t_x), dim=1))

        # loss_final
        loss = BCE_loss + args.lbd_pred*pred_loss + args.lbd_embd*embd_loss
        loss.backward()
        s_optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    return total_loss / total_examples

@torch.no_grad()
def test(model):
    """
    test the student on training dataset, validation dataset, test dataset.
    
    :param model: trained student model
    :return: ROC_AUC on training data, ROC_AUC on validation data, ROC_AUC on test data
    """
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in test_loader:
        data = data.to(device)
        out, _ = model(data.x, data.edge_index, data.edge_attr)

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
t_max_idx = 0
s_max_idx = 0
t_max_acc = [0, 0, 0]
s_max_acc = [0, 0, 0]

t_model.load_state_dict(torch.load(t_PATH))
t_model.eval()
for epoch in range(1, 1001):
    if(epoch==1):
        t_max_acc = test(t_model)
        print(f'[Teacher] Train: {t_max_acc[0]:.4f}, '
            f'Val: {t_max_acc[1]:.4f}, Test: {t_max_acc[2]:.4f}')

    loss = train_student_model(t_model, epoch)
    train_rocauc, valid_rocauc, test_rocauc = test(s_model)
    if(valid_rocauc>s_max_acc[1]):
        s_max_acc[0] = train_rocauc
        s_max_acc[1] = valid_rocauc
        s_max_acc[2] = test_rocauc
        torch.save(s_model.state_dict(), s_PATH)
    print(f'[Student] Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
          f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')

print(f'[Teacher] #Parameters: {count_parameters(t_model):02d}, Train: {t_max_acc[0]:.4f}, '
    f'Val: {t_max_acc[1]:.4f}, Test: {t_max_acc[2]:.4f}')
print(f'[Student] #Parameters: {count_parameters(s_model):02d}, Train: {s_max_acc[0]:.4f}, '
    f'Val: {s_max_acc[1]:.4f}, Test: {s_max_acc[2]:.4f}')
