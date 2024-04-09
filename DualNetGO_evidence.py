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
import os
from sklearn.preprocessing import minmax_scale
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from model_all import *
import pickle
import itertools
import torch_sparse
import warnings
from tqdm import tqdm
import csv

import aslloss
from validation import evaluate_performance

#from comet_ml import Experiment #uncomment this if needed

warnings.filterwarnings("ignore") #temporary ignoring warning from torch_sparse
# Training settings
example_usage = 'python DualNetGO_evidence.py --org human --aspect P --evidence textmining --step1_iter 500 --step2_iter 50 --epochs 100 --max_feat_select 4 --num_adj 5'

parser = argparse.ArgumentParser(description='DualNetGO main function', epilog=example_usage)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--org', type=str, default='human', help='human or mouse')
parser.add_argument('--aspect', type=str, default='P', help='function category')
parser.add_argument('--evidence', type=str, default='neighborhood', help='what kind of evidence to explore')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=512, help='hidden dimensions.')
parser.add_argument('--dropout1', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout2', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout3', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--layer_norm',type=int, default=1, help='layer norm')
parser.add_argument('--w_fc3',type=float, default=0.000, help='Weight decay layer-2')
parser.add_argument('--w_fc2',type=float, default=0.000, help='Weight decay layer-2')
parser.add_argument('--w_fc1',type=float, default=0.000, help='Weight decay layer-1')
parser.add_argument('--lr_fc1',type=float, default=0.01, help='Learning rate 2 fully connected layers')
parser.add_argument('--lr_fc2',type=float, default=0.01, help='Learning rate 2 fully connected layers')
parser.add_argument('--lr_sel',type=float, default=0.01, help='Learning rate for selector')
parser.add_argument('--wd_sel',type=float,default=1e-05,help='weight decay selector layer')
parser.add_argument('--step1_iter',type=int, default=400, help='Step-1 iterations')
parser.add_argument('--step2_iter',type=int, default=20, help='Step-2 iterations')
parser.add_argument('--max_feat_select',type=int, default=5, help='Maximum feature matrices that can be selected.')
parser.add_argument('--num_adj',type=int, default=4, help='Number of dense graph embedding matrices as input')
parser.add_argument('--modeldir',type=str, default='human_evidence', help='folder to save the trained model')
parser.add_argument('--resultdir',type=str, default='.', help='folder to save the csv result')
parser.add_argument('--out',type=str, default='results_evidence.csv', help='csv result')
parser.add_argument('--comet', action='store_true', default=False, help='use comet_ml to log results')


args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

os.makedirs(args.modeldir, exist_ok=True)
os.makedirs(args.resultdir, exist_ok=True)

#maximum length of subset to find
feat_select = int(args.max_feat_select)
sec_iter = args.step2_iter

layer_norm = bool(int(args.layer_norm))

if args.comet:
    with open('/home/zhuoyang/comet_API_token','r') as f:
        experiment = Experiment(f.read().rstrip(), project_name="DualNetGO_evidence")
        experiment.set_name(f'{args.org}_iter1_{args.step1_iter}_iter2_{args.step2_iter}_feat_{args.max_feat_select}_epoch{args.epochs}_{args.aspect}_{args.evidence}_seed{args.seed}')
else :
    experiment=None

print("==========================")
#print(f"Dataset: {args.data}")
#print(f"Dropout1:{args.dropout1}, Dropout2:{args.dropout2}, Dropout3:{args.dropout3}, layer_norm: {layer_norm}")
#print(f" w_fc2:{args.w_fc2}, w_fc1:{args.w_fc1}, w_sel:{args.wd_sel}, lr_fc1:{args.lr_fc1}, lr_fc2:{args.lr_fc2},lr_sel:{args.lr_sel}, 1st step iter: {args.step1_iter}, 2nd step iter: {args.step2_iter}")


criterion = aslloss.AsymmetricLossOptimized(
        gamma_neg=2, gamma_pos=0,
        clip=0.0,
        disable_torch_grad_focal_loss=False,
        eps=1e-5,
    )

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
checkpt_file = f'{args.modeldir}/{args.org}_iter1_{args.step1_iter}_iter2_{args.step2_iter}_feat_{args.max_feat_select}_epoch{args.epochs}_{args.aspect}_{args.evidence}_seed{args.seed}.pt'

if os.path.exists(checkpt_file):
    print('Model exists!')
    import sys
    sys.exit(0)

#set number of adjacency matrices in the input data
num_adj = int(args.num_adj)

def scipy_to_tensor(mat):
    mat = mat.tocoo()
    values = torch.FloatTensor(mat.data)
    #row = torch.LongTensor(mat.row)
    #col = torch.LongTensor(mat.col)
    indices = np.vstack((mat.row, mat.col))
    indices = torch.LongTensor(indices)
    shape = mat.shape

    return torch.sparse.FloatTensor(indices, values, torch.Size(shape))


def train_step(model,optimizer,labels,list_mat,list_ind):
    model.train()
    optimizer.zero_grad()
    output = model(list_mat, layer_norm,list_ind)
    outs = nn.functional.sigmoid(output.detach()).cpu().numpy()
    perf_train = evaluate_performance(labels, outs, (outs > 0.5).astype(int))
    loss_train = criterion(torch.zeros(2,2,2), output, torch.tensor(labels).to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),perf_train['Fmax']


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
        output = model(list_mat, layer_norm,list_ind)
        outs = nn.functional.sigmoid(output.detach()).cpu().numpy()
        loss_test = criterion(torch.zeros(2,2,2), output, torch.tensor(labels).to(device))
        perf_test = evaluate_performance(labels, outs, (outs > threshold).astype(int))
        #print(mask_val)
        return loss_test.item(),perf_test



def selector_step(model,optimizer_sel,mask,o_loss):
    model.train()
    optimizer_sel.zero_grad()
    mask.requires_grad = True
    output = model(mask,o_loss)
    selector_loss = 10*F.mse_loss(output,o_loss)
    selector_loss.backward()
    input_grad = mask.grad.data
    optimizer_sel.step()
    return selector_loss.item(), input_grad


def selector_eval(model,mask,o_loss):
    model.eval()
    with torch.no_grad():
        output = model(mask,o_loss)
        selector_loss = F.mse_loss(output,o_loss)
        return selector_loss.item()


def new_optimal_mask(model, model_sel, optimizer_sel, list_val_mat, device, labels, num_layer):

    #Calculate input gradients
    equal_masks = torch.ones(num_layer).float().to(device)
    #Assign same weight to all indices
    equal_masks *= 0.5
    model_sel.train()
    optimizer_sel.zero_grad()
    equal_masks.requires_grad = True
    output = model_sel(equal_masks,None)
    output.backward()
    tmp_grad = equal_masks.grad.data
    tmp_grad = torch.abs(tmp_grad)

    #Top mask indices by gradients
    best_grad = sorted(torch.argsort(tmp_grad)[-feat_select:].tolist())

    #Creating possible optimal subsets with top mask indices
    new_combinations = list()
    for ll in range(1,feat_select+1):
        new_combinations.extend(list(itertools.combinations(best_grad,ll)))

    list_ind = list(range(len(new_combinations)))

    best_mask = []
    best_mask_loss = []
    #From these possible subsets, sample and check validation loss
    for _ in range(10):
        get_ind = random.choices(list_ind)[0]
        get_ind = list(new_combinations[get_ind])
        get_ind = sorted(get_ind)
        best_mask.append(get_ind)
        input_val_mat = [list_val_mat[ww] for ww in get_ind]

        loss_val,acc_val,_ = validate_step(model,labels,input_val_mat,get_ind)
        best_mask_loss.append(loss_val)


    #Find indices with minimum validation loss
    min_loss_ind = np.argmin(best_mask_loss)
    optimal_mask = best_mask[min_loss_ind]


    return optimal_mask, model_sel, model



def train(list_train_mat,list_val_mat,list_test_mat,list_label,num_nodes,num_feat,num_labels):

    list_train_mat = [mat.to(device) for mat in list_train_mat]
    list_val_mat = [mat.to(device) for mat in list_val_mat]

    #Set number of linear layers of input adj/feat to create
    num_adj_mat = num_adj
    num_feat_mat = len(list_train_mat) - num_adj

    num_layer = len(list_train_mat)
    model = Classifier_evidence(nfeat=num_feat,
                nhidden=args.hidden,
                nclass=num_labels,
                dropout1=args.dropout1,
                dropout2=args.dropout2,
                dropout3=args.dropout3,
                num_nodes=num_nodes, device=int(args.dev)).to(device)


    optimizer_sett_classifier = [
        {'params': model.fc2.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc2},
        {'params': model.fc3.parameters(), 'weight_decay': args.w_fc3, 'lr': args.lr_fc2},
        {'params': model.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc1},
    ]

    optimizer = optim.Adam(optimizer_sett_classifier)

    model_sel = Selector(num_layer,256).to(device)
    optimizer_select = [
        {'params':model_sel.fc1.parameters(), 'weight_decay':args.wd_sel, 'lr':args.lr_sel},
        {'params':model_sel.fc2.parameters(), 'weight_decay':args.wd_sel, 'lr':args.lr_sel}
    ]
    optimizer_sel = optim.Adam(optimizer_select)

    bad_counter = 0
    best = 999999999
    best_sub = []


    #Calculate all possible combinations of subsets upto length feat_select
    combinations = list()
    for nn in range(1,feat_select+1):
        combinations.extend(list(itertools.combinations(range(num_layer),nn)))


    dict_comb = dict()
    for kk,cc in enumerate(combinations):
        dict_comb[cc] = kk

    #Step-1 training: Exploration step
    print('Step1: Exploration===============')
    for epoch in tqdm(range(args.step1_iter)):
        #choose one subset randomly
        rand_ind = random.choice(combinations)
        #create input to model
        input_train_mat = [list_train_mat[ww] for ww in rand_ind]
        input_val_mat = [list_val_mat[ww] for ww in rand_ind]

        #Train classifier and selector
        loss_tra,acc_tra = train_step(model,optimizer,list_label[0],input_train_mat,rand_ind)
        loss_val,acc_val,_ = validate_step(model,list_label[1],input_val_mat,rand_ind)

        #Input mask vector to selector
        input_mask = torch.zeros(num_layer).float().to(device)
        input_mask[list(rand_ind)] = 1.0
        input_loss = torch.FloatTensor([loss_tra]).to(device)
        eval_loss = torch.FloatTensor([loss_val]).to(device)
        loss_select, input_grad = selector_step(model_sel,optimizer_sel,input_mask,input_loss)
        #loss_select_val = selector_eval(model_sel,input_mask,eval_loss)

        #log metrics
        if args.comet:
            experiment.log_metric('train_loss', loss_tra, epoch=epoch+1)
            experiment.log_metric('train_Fmax', acc_tra, epoch=epoch+1)
            experiment.log_metric('valid_loss', loss_val, epoch=epoch+1)
            experiment.log_metric('valid_Fmax', acc_val, epoch=epoch+1)


    #Starting Step-2: Exploitation
    print('Step2: Exploitation===============')
    dict_check_loss = dict()
    for epoch in tqdm(range(args.epochs)):

        if epoch<sec_iter:
            #Up to sec_iter epoches optimal subsets are identified
            train_mask, model_sel, model = new_optimal_mask(model, model_sel, optimizer_sel, list_val_mat,device, list_label[1],num_layer)


        if epoch==sec_iter:
            min_ind = min(list(dict_check_loss.keys()))
            train_mask = dict_check_loss[min_ind]


        input_train_mat = [list_train_mat[ww] for ww in train_mask]
        input_val_mat = [list_val_mat[ww] for ww in train_mask]
        loss_tra,acc_tra = train_step(model,optimizer,list_label[0],input_train_mat,train_mask)
        loss_val,acc_val,tmax = validate_step(model,list_label[1],input_val_mat,train_mask)

        if args.comet:
            experiment.log_metric('train_loss', loss_tra, epoch=epoch+1+args.step1_iter)
            experiment.log_metric('train_Fmax', acc_tra, epoch=epoch+1+args.step1_iter)
            experiment.log_metric('valid_loss', loss_val, epoch=epoch+1+args.step1_iter)
            experiment.log_metric('valid_Fmax', acc_val, epoch=epoch+1+args.step1_iter)


        dict_check_loss[loss_val] = train_mask

        if epoch < sec_iter:
            input_mask = torch.zeros(num_layer).float().to(device)
            input_mask[list(train_mask)] = 1.0
            input_loss = torch.FloatTensor([loss_tra]).to(device)
            eval_loss = torch.FloatTensor([loss_val]).to(device)
            loss_select, _ = selector_step(model_sel,optimizer_sel,input_mask,input_loss)
            #loss_select_val = selector_eval(model_sel,input_mask,eval_loss)


        if loss_val < best and epoch>= sec_iter:
            best = loss_val
            #save model
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
            best_sub = train_mask

        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break


    select_ind = best_sub

    del list_train_mat
    del list_val_mat
    input_test_mat = [list_test_mat[ww] for ww in select_ind]

    test_out = test_step(model,list_label[2],input_test_mat,select_ind,tmax)
    perf = test_out[1]

    if args.comet:
        experiment.log_metric('test_Fmax', perf['Fmax'], step=1)
        experiment.log_metric('test_acc', perf['acc'], step=1)
        experiment.log_metric('test_F1', perf['F1'], step=1)
        experiment.log_metric('test_m-aupr', perf['m-aupr'], step=1)
        experiment.log_metric('test_M-aupr', perf['M-aupr'], step=1)

    return perf, select_ind



## main process =====================================================
acc_list = []
Annot = sio.loadmat(f'data/{args.org}/{args.org}_annot.mat', squeeze_me=True)
train_idx = Annot['indx'][args.aspect].tolist()['train'].tolist().tolist()
valid_idx = Annot['indx'][args.aspect].tolist()['valid'].tolist().tolist()
test_idx = Annot['indx'][args.aspect].tolist()['test'].tolist().tolist()

labels_train = np.array(Annot['GO'][args.aspect].tolist()['train'].tolist())
labels_valid = np.array(Annot['GO'][args.aspect].tolist()['valid'].tolist())
labels_test = np.array(Annot['GO'][args.aspect].tolist()['test'].tolist())


# load adj_mat or embedding:
# embeddings
list_mat = []
for n in ['node2vec','GAE','MLPAE','AE']:
    fn = f'data/{args.org}/{args.org}_net_{args.evidence}_{n}.npy'
    y = np.load(fn)
    list_mat.append(y)
del y


fn = f'data/{args.org}/{args.org}_net_{args.evidence}.mat'
y = sio.loadmat(fn, squeeze_me=True)
y = y['Net'].todense()
y = minmax_scale(np.asarray(y))
list_mat.append(y)
del y


# load features
if args.org == 'mouse':
    with open(f'data/{args.org}/features_mouse.npy', 'rb') as f:
        Z = pickle.load(f)
else:
    with open(f'data/{args.org}/features.npy', 'rb') as f:
        Z = pickle.load(f)
list_mat.append(Z)
del Z

num_nodes = list_mat[-2].shape[0]

num_labels = labels_train.shape[1]
num_feat = list_mat[-1].shape[1]

t_total = time.time()
list_total_acc = []
for i in range(1):

    #Create training and testing split
    list_train_mat  = []
    list_val_mat = []
    list_test_mat = []

    for mat in list_mat:
        mat = torch.from_numpy(mat).float()
        list_train_mat.append(mat[train_idx,:])
        list_val_mat.append(mat[valid_idx,:])
        list_test_mat.append(mat[test_idx,:])

    list_label = [labels_train, labels_valid, labels_test]

    del mat

    accuracy_data, best_mask = train(list_train_mat,list_val_mat,list_test_mat,list_label,num_nodes,num_feat,num_labels)
    num_layer = len(list_train_mat)

    acc_list.append(accuracy_data['Fmax'])
    list_total_acc.append(100-accuracy_data['Fmax'])


training_time = time.time() - t_total
if args.comet:
    experiment.log_metric('training_time', "{:.4f}s".format(training_time), step=1)

print("Train time: {:.4f}s".format(training_time))
print(f"Test accuracy: {np.mean(acc_list):.2f}, {np.round(np.std(acc_list),2)}")

fn = f'{args.resultdir}/{args.out}'
with open(fn, 'a') as f:
    csv.writer(f).writerow([args.org, args.w_fc1, args.w_fc2, args.w_fc3, args.dropout1, args.dropout2, args.dropout3, args.lr_fc1, args.lr_fc2, args.lr_sel, args.wd_sel, args.hidden, args.layer_norm, args.step1_iter, args.step2_iter, args.epochs, args.max_feat_select, args.num_adj, args.aspect, args.evidence, accuracy_data['Fmax'], accuracy_data['F1'], accuracy_data['M-F1'], accuracy_data['acc'], accuracy_data['m-aupr'], accuracy_data['M-aupr'], best_mask])
