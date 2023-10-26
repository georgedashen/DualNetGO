import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import random
import scipy.io as sio
from tqdm import tqdm
import csv
import time

import aslloss
from validation import evaluate_performance

class DNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(1600, 1024)
        self.w2 = nn.Linear(1024, 512)
        self.w3 = nn.Linear(512, 256)
        self.w4 = nn.Linear(256, args.num_class)

        self.norm1 = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(256)
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        out = self.w1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.w2(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.w3(out)
        out = self.norm3(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.w4(out)

        return out

def set_rand_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_step(model, X, labels):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    #acc_train = accuracy(output, labels)
    #loss_train = F.nll_loss(output, labels.to(device))
    outs = nn.functional.sigmoid(output.detach()).cpu().numpy()
    perf_train = evaluate_performance(labels, outs, (outs > 0.5).astype(int))
    loss_train = criterion(torch.zeros(2,2,2), output, torch.tensor(labels).to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),perf_train['Fmax']


def validate_step(model, X, labels):
    model.eval()
    with torch.no_grad():
        output = model(X)
        #loss_val = F.nll_loss(output, labels.to(device))
        #acc_val = accuracy(output, labels)
        outs = nn.functional.sigmoid(output.detach()).cpu().numpy()
        loss_val = criterion(torch.zeros(2,2,2), output, torch.tensor(labels).to(device))
        perf_val = evaluate_performance(labels, outs, (outs > 0.5).astype(int))
        return loss_val.item(),perf_val['Fmax'],perf_val['tmax']

def test_step(model, X, labels, threshold):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(X)
        #loss_test = F.nll_loss(output, labels.to(device))
        #acc_test = accuracy(output, labels)
        outs = nn.functional.sigmoid(output.detach()).cpu().numpy()
        loss_test = criterion(torch.zeros(2,2,2), output, torch.tensor(labels).to(device))
        perf_test = evaluate_performance(labels, outs, (outs > threshold).astype(int))
        #print(mask_val)
        return loss_test.item(),perf_test

parser = argparse.ArgumentParser()
parser.add_argument('--org', type=str, default='human')
parser.add_argument('--aspect', type=str, default='P')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--bach_size', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--outdir', type=str, default='.')
parser.add_argument('--out', type=str, default='results.csv')
args = parser.parse_args()

set_rand_seed(args.seed)
device = torch.device('cuda:0')

x1 = np.load(f'{args.org}_net_adj_VGAE_epoch100.npy')
x2 = np.load(f'{args.org}_net_sim_VGAE_epoch100.npy')

X = torch.cat([torch.tensor(x1).float(), torch.tensor(x2).float()],1)

Annot = sio.loadmat(f'../data/{args.org}/{args.org}_annot.mat', squeeze_me=True)
train_idx = Annot['indx'][args.aspect].tolist()['train'].tolist().tolist()
valid_idx = Annot['indx'][args.aspect].tolist()['valid'].tolist().tolist()
test_idx = Annot['indx'][args.aspect].tolist()['test'].tolist().tolist()

labels_train = np.array(Annot['GO'][args.aspect].tolist()['train'].tolist())
labels_valid = np.array(Annot['GO'][args.aspect].tolist()['valid'].tolist())
labels_test = np.array(Annot['GO'][args.aspect].tolist()['test'].tolist())

X_train = X[train_idx, :]
X_valid = X[valid_idx, :]
X_test = X[test_idx, :]

args.num_class = labels_train.shape[1]

model = DNN(args).to(device)

optimizer_sett_classifier = [
        {'params': model.w1.parameters(), 'weight_decay': 0.01, 'lr': args.lr},
        {'params': model.w2.parameters(), 'weight_decay': 0.01, 'lr': args.lr},
        {'params': model.w3.parameters(), 'weight_decay': 0.01, 'lr': args.lr},
        {'params': model.w4.parameters(), 'weight_decay': 0.01, 'lr': args.lr}
    ]

optimizer = optim.Adam(optimizer_sett_classifier)

criterion = aslloss.AsymmetricLossOptimized(
        gamma_neg=2, gamma_pos=0,
        clip=0.0,
        disable_torch_grad_focal_loss=False,
        eps=1e-5,
    )

checkpt_file = f'{args.outdir}/{args.org}_{args.aspect}_seed{args.seed}_model.pkl'
best_Fmax = 0
t0 = time.time()
for epoch in tqdm(range(args.epochs)):
    train_loss, train_Fmax = train_step(model, X_train.to(device), labels_train)
    print(f'Training Fmax: {train_Fmax}, Loss: {train_loss}')

    if epoch >= 50:
        valid_loss, valid_Fmax, tmax = validate_step(model, X_valid.to(device), labels_valid)
        if valid_Fmax > best_Fmax:
            best_Fmax = valid_Fmax
            torch.save(model.state_dict(), checkpt_file)
            test_loss, test_perf = test_step(model, X_test.to(device), labels_test, tmax)
print(f'Training time: {time.time()-t0}')

with open(args.out, 'a') as f:
    csv.writer(f).writerow([args.org, args.aspect, test_perf['Fmax'], test_perf['F1'], test_perf['M-F1'], test_perf['acc'], test_perf['m-aupr'], test_perf['M-aupr']])
