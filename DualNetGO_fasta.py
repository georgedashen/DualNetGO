##############################################
## Revised by Zhuoyang CHEN
## Date: 2023-07-17
#############################################

from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
from scipy import sparse
import scipy.io as sio
from sklearn.preprocessing import minmax_scale
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import *
from model import *
import uuid
import pickle
import copy
import itertools
from collections import defaultdict
import warnings
from tqdm import tqdm
import pandas as pd
import sys
import os
import ast
import csv

import aslloss
from validation import evaluate_performance

warnings.filterwarnings("ignore") #temporary ignoring warning from torch_sparse
# Training settings
example_usage = 'CUDA_VISIBLE_DEVICES=0 python DualNetGO_fasta.py --fasta ./data/human_C_test.fa --org human --aspect C --checkpoint ./human_best/iter1_500_iter2_80_feat_3_epoch100_C_AE_seed42.pt --out prediction.csv'

example_usage2 = 'CUDA_VISIBLE_DEVICES=0 python DualNetGO_fasta.py --txt ./query_result_human.txt --org human --aspect C --checkpoint ./human_best/iter1_500_iter2_80_feat_3_epoch100_C_AE_seed42.pt --out prediction.csv'

parser = argparse.ArgumentParser(description='DualNetGO test given model', epilog=example_usage)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--org', type=str, default='human', help='which species to test')
parser.add_argument('--aspect', type=str, default='P', help='function category')
parser.add_argument('--fasta', type=str, default='', help='fasta file input')
parser.add_argument('--txt', type=str, default='', help='blast result txt file as input')
parser.add_argument('--checkpoint',type=str, default='', help='explicitly provide model checkpoint file')
parser.add_argument('--embedding', type=str, default='AE', help='what kind of embedding to use')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--layer', type=int, default=3, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=512, help='hidden dimensions.')
parser.add_argument('--dropout1', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout2', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout3', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--layer_norm',type=int, default=1, help='layer norm')
parser.add_argument('--w_fc3',type=float, default=0.000, help='Weight decay layer-2')
parser.add_argument('--w_fc2',type=float, default=0.000, help='Weight decay layer-2')
parser.add_argument('--w_fc1',type=float, default=0.000, help='Weight decay layer-1')
parser.add_argument('--lr_fc1',type=float, default=0.01, help='Learning rate 2 fully connected layers')
parser.add_argument('--lr_fc2',type=float, default=0.01, help='Learning rate 2 fully connected layers')
parser.add_argument('--lr_sel',type=float, default=0.01, help='Learning rate for selector')
parser.add_argument('--wd_sel',type=float,default=1e-05,help='weight decay selector layer')
parser.add_argument('--step1_iter',type=int, default=100, help='Step-1 iterations')
parser.add_argument('--step2_iter',type=int, default=30, help='Step-2 iterations')
parser.add_argument('--directed',type=int, default=0, help='Undirected:0, Directed:1')
parser.add_argument('--max_feat_select',type=int, default=5, help='Maximum feature matrices that can be selected.')
parser.add_argument('--num_adj',type=int, default=7, help='Number of sparse adjacency matrices(including powers) as input')
parser.add_argument('--modeldir',type=str, default='best', help='folder to save the trained model')
parser.add_argument('--resultdir',type=str, default='.', help='folder to save the csv result')
parser.add_argument('--out',type=str, default='prediction.csv', help='csv result')


args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


#maximum length of subset to find
feat_select = int(args.max_feat_select)
sec_iter = args.step2_iter

layer_norm = bool(int(args.layer_norm))

print("==========================")

criterion = aslloss.AsymmetricLossOptimized(
        gamma_neg=2, gamma_pos=0,
        clip=0.0,
        disable_torch_grad_focal_loss=False,
        eps=1e-5,
    )

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)

if args.checkpoint == '':
    checkpt_file = f'{args.modeldir}/iter1_{args.step1_iter}_iter2_{args.step2_iter}_feat_{args.max_feat_select}_epoch{args.epochs}_{args.aspect}_{args.embedding}_seed{args.seed}.pt'
elif os.path.isfile(args.checkpoint):
    checkpt_file = args.checkpoint
    temp_args = os.path.split(checkpt_file)[1].split('_')
    args.step1_iter = int(temp_args[1])
    args.step2_iter = int(temp_args[3])
    args.max_feat_select = int(temp_args[5])
    args.aspect = temp_args[7]
    args.embedding = temp_args[8]

#set number of adjacency matrices in the input data
is_directed = bool(args.directed)
num_adj = int(args.num_adj)


def validate_step(model,labels,list_mat,list_ind):
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm,list_ind)
        outs = nn.functional.sigmoid(output.detach()).cpu().numpy()
        loss_val = criterion(torch.zeros(2,2,2), output, torch.tensor(labels).to(device))
        perf_val = evaluate_performance(labels, outs, (outs > 0.5).astype(int))
        return loss_val.item(),perf_val['Fmax'],perf_val['tmax']




def test_step(model,labels,list_mat,list_ind,threshold):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm, list_ind)
        outs = nn.functional.sigmoid(output.detach()).cpu().numpy()
        loss_test = criterion(torch.zeros(2,2,2), output, torch.tensor(labels).to(device))
        perf_test = evaluate_performance(labels, outs, (outs > threshold).astype(int))
        return loss_test.item(),perf_test




def test(list_val_mat,list_test_mat,list_label,num_nodes,num_feat,num_labels):
  
    list_val_mat = [mat.to(device) for mat in list_val_mat]

    #Set number of linear layers of input adj/feat to create
    num_adj_mat = num_adj
    num_feat_mat = len(list_train_mat) - num_adj

    num_layer = len(list_train_mat)
    model = Classifier(nfeat=num_feat,
                num_adj_mat=num_adj_mat,
                num_feat_mat=num_feat_mat,
                nhidden=args.hidden,
                nclass=num_labels,
                dropout1=args.dropout1,
                dropout2=args.dropout2,
                dropout3=args.dropout3,
                num_nodes=num_nodes, device=int(args.dev)).to(device)


    print('Testing: =======================')
    input_val_mat = [list_val_mat[ww] for ww in best_mask]
    input_test_mat = [list_test_mat[ww] for ww in best_mask]
    loss_val,acc_val,tmax = validate_step(model,list_label[1],input_val_mat,best_mask)

    test_out = test_step(model,list_label[2],input_test_mat,best_mask,tmax)
    perf = test_out[1]

    return perf, best_mask



def predict(list_test_mat,num_nodes,num_feat,num_labels):
  
    list_test_mat = [mat.to(device) for mat in list_test_mat]

    #Set number of linear layers of input adj/feat to create
    num_adj_mat = num_adj
    num_feat_mat = len(list_test_mat) - num_adj

    num_layer = len(list_test_mat)
    model = Classifier(nfeat=num_feat,
                num_adj_mat=num_adj_mat,
                num_feat_mat=num_feat_mat,
                nhidden=args.hidden,
                nclass=num_labels,
                dropout1=args.dropout1,
                dropout2=args.dropout2,
                dropout3=args.dropout3,
                num_nodes=num_nodes, device=int(args.dev)).to(device)


    print('Predicting: =======================')
    input_test_mat = [list_test_mat[ww] for ww in best_mask]
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(input_test_mat, layer_norm, best_mask)
        outs = nn.functional.sigmoid(output.detach()).cpu().numpy()

    return outs

## main process =====================================================
t_total = time.time()

Annot = sio.loadmat(f'data/{args.org}_annot.mat', squeeze_me=True)
#train_idx = Annot['indx'][args.aspect].tolist()['train'].tolist().tolist()
valid_idx = Annot['indx'][args.aspect].tolist()['valid'].tolist().tolist()
#test_idx = Annot['indx'][args.aspect].tolist()['test'].tolist().tolist()

## blast retrieve test_idx
if args.fasta:
    cmd_db = f'makeblastdb -in ./data/{args.org}.fa -dbtype prot -out {args.org}_prot_set'
    if os.path.isfile(f'{args.org}_prot_set.phr') and os.path.isfile(f'{args.org}_prot_set.pin') and os.path.isfile(f'{args.org}_prot_set.psq'):
        pass
    else:
        os.system(cmd_db)

    cmd_blast = f'blastp -db {args.org}_prot_set -query {args.fasta} -out query_result_{args.org}.txt -outfmt 6 -max_target_seqs 1'
    os.system(cmd_blast)

if args.txt:
    fRead = open(args.txt, 'r')
else:
    fRead = open(f'query_result_{args.org}.txt', 'r')
query_id = []
target_id = []
for i, line in tqdm(enumerate(fRead)):
    splitted = line.strip().split('\t')
    query_id.append(splitted[0])
    target_id.append(splitted[1])

protein_id = np.genfromtxt(f'./data/{args.org}_protein_id.txt', dtype=str, skip_header=1)
test_idx = [np.where(protein_id == i)[0][0] for i in target_id]

#labels_train = np.array(Annot['GO'][args.aspect].tolist()['train'].tolist())
#labels_valid = np.array(Annot['GO'][args.aspect].tolist()['valid'].tolist())
#labels_test = np.array(Annot['GO'][args.aspect].tolist()['test'].tolist())

GO_term = Annot['labels'][args.aspect].tolist().tolist()[0]


# load adj_mat or embedding:
# embeddings
list_mat = []
if args.embedding != 'None':
    for e in ['neighborhood', 'fusion', 'cooccurence', 'coexpression', 'experimental', 'database', 'textmining']:
        fn = f'data/{args.org}_net_{e}_{args.embedding}.npy'
        y = np.load(fn)
        list_mat.append(y)
    del y
else:
# adj_mat
    for e in ['neighborhood', 'fusion', 'cooccurence', 'coexpression', 'experimental', 'database', 'textmining']:
        fn = f'data/{args.org}_net_{e}.mat'
        y = sio.loadmat(fn, squeeze_me=True)
        y = y['Net'].todense()
        y = minmax_scale(y)
        list_mat.append(y)
    del y


# load features
if args.org == 'mouse':
    with open(f'data/features_mouse.npy', 'rb') as f: #for human, you may change the feature input as 'data/human/features.npy' or make a copy as 'data/human/features_human.npy'
        Z = pickle.load(f)
elif args.org == 'human':
    with open(f'data/features.npy', 'rb') as f:
        Z = pickle.load(f)
        
list_mat.append(Z)
del Z


# load results for best mask
res = pd.read_csv(f'./results_{args.org}_best.csv', header=None) #please ensure to run DualNetGO training as least once to attain a record in the csv output, and corresponding checkpoint file
df = res[(res[12]==args.step1_iter) & (res[13]==args.step2_iter) & (res[15]==args.max_feat_select) & (res[17]==args.aspect) & ((res[18]==args.embedding))]

if len(df) == 0 or (not os.path.isfile(checkpt_file)):
    print('Model result not found!')
    sys.exit(0)

best_mask = ast.literal_eval(df[25].values[0])

num_nodes = list_mat[-2].shape[1] # mat
num_labels = GO_term.shape[0] # y
num_feat = list_mat[-1].shape[1] # features


#Create training and testing split
#list_train_mat  = []
#list_val_mat = []
list_test_mat = []

for mat in list_mat:
    mat = torch.from_numpy(mat).float()
    #list_train_mat.append(mat[train_idx,:])
    #list_val_mat.append(mat[valid_idx,:])
    list_test_mat.append(mat[test_idx,:])

del mat

scores = predict(list_test_mat,num_nodes,num_feat,num_labels)

df_result = pd.DataFrame(scores,columns=GO_term)
df_result['id'] = query_id
dfm = df_result.melt(id_vars='id',value_vars=GO_term,ignore_index=True)
dfm.to_csv(f'{args.out}', index=False)



