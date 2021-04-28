from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import *
import uuid

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of teacher layers.')
parser.add_argument('--t_hidden', type=int, default=64, help='teacher hidden dimensions.')
parser.add_argument('--s_hidden', type=int, default=64, help='student hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
parser.add_argument('--lbd_pred', type=float, default=0, help='lambda for prediction loss')
parser.add_argument('--lbd_embd', type=float, default=0, help='lambda for embedding loss')
parser.add_argument('--kernel', default='kl', help='kernel functions: kl,lin,poly,dist,RBF')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.data)
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
t_PATH = "./src/citation/teacher/teacher_"+str(args.data)+str(args.layer)+".pth"
checkpt_file = "./src/citation/student/student_"+str(args.data)+str(args.layer)+".pth"

# Define model
teacher = GCNII(nfeat=features.shape[1],
                        nlayers=args.layer,
                        nhidden=args.t_hidden,
                        nclass=int(labels.max()) + 1,
                        dropout=args.dropout,
                        lamda = args.lamda,
                        alpha=args.alpha,
                        variant=args.variant).to(device)
teacher.load_state_dict(torch.load(t_PATH))
model = GCNII_student(nfeat=features.shape[1],
                nlayers=args.layer,
                thidden=args.t_hidden,
                nhidden=args.s_hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                lamda = args.lamda,
                alpha=args.alpha,
                variant=args.variant).to(device)

optimizer = optim.Adam([
                        {'params':model.params1,'weight_decay':args.wd1},
                        {'params':model.params2,'weight_decay':args.wd2},
                        ],lr=args.lr)

temperature = 2
def train():
    """
    Start training with a stored hyperparameters on the dataset
    :make sure teacher, student, optimizer, node features, adjacency, train index is defined aforehead
    :return: train loss, train accuracy
    """
    teacher.eval()
    model.train()
    optimizer.zero_grad()

    # loss_CE
    t_output, t_hidden = teacher(features,adj)
    s_output, s_hidden = model(features,adj)
    s_out = F.log_softmax(s_output, dim=1)
    loss_CE = F.nll_loss(s_out[idx_train], labels[idx_train].to(device))
    acc_train = accuracy(s_out[idx_train], labels[idx_train].to(device))

    # loss_task
    t_output = t_output/temperature
    t_y = t_output[idx_train]
    s_y = s_output[idx_train]
    loss_task = kernel(t_y, s_y, 'kl')

    # loss_hidden
    t_x = t_hidden[idx_train]
    s_x = s_hidden[idx_train]
    loss_hidden = kernel(t_x, s_x, args.kernel)

    # loss_final
    loss_train = loss_CE + args.lbd_pred*loss_task + args.lbd_embd*loss_hidden
    loss_train.backward()
    optimizer.step()

    return loss_train.item(),acc_train.item()

def validate():
    """
    Validate the model
    make sure teacher, student, optimizer, node features, adjacency, validation index is defined aforehead
    :return: validation loss, validation accuracy
    """
    teacher.eval()
    model.eval()

    with torch.no_grad():
        # loss_CE
        t_output, t_hidden = teacher(features,adj)
        s_output, s_hidden = model(features,adj)
        s_out = F.log_softmax(s_output, dim=1)
        loss_CE = F.nll_loss(s_out[idx_val], labels[idx_val].to(device))

        # loss_task
        t_y = t_output[idx_val]
        s_y = s_output[idx_val]
        loss_task = kernel(t_y, s_y, 'kl')

        # loss_hidden
        t_x = t_hidden[idx_val]
        s_x = s_hidden[idx_val]
        loss_hidden = kernel(t_x, s_x, args.kernel)

        # loss_final
        loss_val = loss_CE + args.lbd_pred*loss_task + args.lbd_embd*loss_hidden
        acc_val = accuracy(s_output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test():
    """
    Test the model
    make sure student, node features, adjacency, test index is defined aforehead
    :return: test accuracy
    """
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output,_ = model(features, adj)
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return acc_test.item()

# Start Training
t_total = time.time()
bad_counter = 0
best = 999999999
best_epoch = 0
acc = 0
for epoch in range(args.epochs):
    loss_tra,acc_tra = train()
    loss_val,acc_val = validate()
    if(epoch+1)%1 == 0:
        print('Epoch:{:04d}'.format(epoch+1),
            'train',
            'loss:{:.3f}'.format(loss_tra),
            'acc:{:.2f}'.format(acc_tra*100),
            '| val',
            'loss:{:.3f}'.format(loss_val),
            'acc:{:.2f}'.format(acc_val*100))
    if loss_val < best:
        best = loss_val
        best_epoch = epoch
        acc = acc_val
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == 200:
        break

if args.test:
    acc = test()

print('The number of parameters in the student: {:04d}'.format(count_params(model)))
print('Load {}th epoch'.format(best_epoch))
print("Test" if args.test else "Val","acc.:{:.2f}".format(acc*100))
