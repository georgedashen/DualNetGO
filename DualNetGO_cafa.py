##############################################
## Revised by Zhuoyang CHEN
## Date: 2024-03-21
## For large dataset cafa3 training and general prediction
#############################################

from __future__ import division
from __future__ import print_function
import time
import os,sys
import random
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
import scipy.io as sio
import ast
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from model import *
from get_dataset import multimodesDataset
import itertools
from collections import defaultdict
import warnings
from tqdm import tqdm
import csv

import aslloss
from validation import evaluate_performance


warnings.filterwarnings("ignore") #temporary ignoring warning from torch_sparse

example_usage = 'CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=6005 DualNetGO_cafa.py --mode train --lr_fc1 0.01 --lr_fc2 0.01 --step1_iter 100 --step2_iter 10 --max_feat_select 3 --aspect C --noeval_in_train --txt cc_query_results.txt'
noparallel_usage = 'CUDA_VISIBLE_DEVICES=0 python DualNetGO_cafa.py --mode train --parallel 0 --batch 100000 --lr_fc1 0.01 --lr_fc2 0.01 --step1_iter 100 --step2_iter 10 --max_feat_select 3 --aspect C --noeval_in_train --txt cc_query_results.txt'
predict_usage_fasta = 'CUDA_VISIBLE_DEVICES=0 python DualNetGO_cafa.py --mode predict --aspect C --fasta cc-test.fasta --resultdir data/cafa3'
predict_usage_txt = 'CUDA_VISIBLE_DEVICES=0 python DualNetGO_cafa.py --mode predict --aspect C --txt cc_query_results.txt --resultdir data/cafa3'
predict_usage_custom = 'CUDA_VISIBLE_DEVICES=0 python DualNetGO_cafa.py --mode predict --aspect C --txt cc_query_results.txt --resultdir data/cafa3 --checkpt {your_model} --resultfile {your_file}'


parser = argparse.ArgumentParser(description='DualNetGO_CAFA main function with mini-batch DDP', epilog=example_usage)

#main settings (input, output)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--org', type=str, default='all', help='train with all species or specify by taxo code')
parser.add_argument('--aspect', type=str, default='P', help='function category')
parser.add_argument('--fasta', type=str, default='', help='fasta input file')
parser.add_argument('--txt', type=str, default='', help='blast results input file')
parser.add_argument('--mode',type=str, choices=['train','predict'],help='train or predict')
parser.add_argument('--embedding', type=str, default='AE', help='what kind of embedding to use')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=128, help='batch size for dataloader')
parser.add_argument('--workers', type=int, default=32)
parser.add_argument('--parallel', type=int, default=1)
parser.add_argument('--modeldir',type=str, default='temp', help='folder to save the trained model')
parser.add_argument('--resultdir',type=str, default='.', help='folder to save the csv result')
parser.add_argument('--out',type=str, default='results.csv', help='csv result')

# model settings
parser.add_argument('--layer', type=int, default=3, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=1024, help='hidden dimensions.')
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
parser.add_argument('--num_adj',type=int, default=7, help='Number of sparse adjacency matrices(including powers) as input')

# training and monitoring
parser.add_argument('--patience', type=int, default=100, help='epochs for early stopping')
parser.add_argument('--saveModel', type=int, default=1, help='whether save trained model')
parser.add_argument('--overwrite', action='store_true', default=True)
parser.add_argument('--comet', action='store_true', default=False, help='use comet_ml to log results')
parser.add_argument('--noeval_in_train', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)

# custom prediction using other checkpoints
parser.add_argument('--checkpt', type=str, default='', help='only loaded in predict mode')
parser.add_argument('--mask',default=None,help='Provide feature mask as list, eg.[0,1]')
parser.add_argument('--resultfile', type=str, default='', help='a csv file that contains training results')

args = parser.parse_args()


# set seed for reproduction
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

os.makedirs(args.resultdir, exist_ok=True)
os.makedirs(args.modeldir, exist_ok=True)

feat_select = int(args.max_feat_select) #maximum length of subset to find
num_adj = int(args.num_adj) #set number of adjacency matrices in the input data
sec_iter = args.step2_iter
if args.mode == 'predict':
    args.parallel = 0
parallel = bool(int(args.parallel))
layer_norm = bool(int(args.layer_norm))
args.saveModel = bool(int(args.saveModel))


# set parallel
if parallel:
    local_rank=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    dist_rank = dist.get_rank()
    #print(f'world_size: {dist.get_world_size()}')
    #print(f'local_rank: {dist.get_rank()}')
else:
    if torch.cuda.is_available():
        #local_rank = torch.device('cuda:0')
        local_rank = 0
    else:
        local_rank = torch.device('cpu')
    dist_rank = 0
device = torch.device('cuda', local_rank)


# set comet_ml, default turn off, make sure to use your own API_token in comet_ml
if args.comet and dist_rank==0:
    from comet_ml import Experiment #comment this if not needed
    with open('/home/zhuoyang/comet_API_token','r') as f:
        experiment = Experiment(f.read().rstrip(), project_name="DualNetGO_cafa3")
        experiment.set_name(f'{args.org}_iter1_{args.step1_iter}_iter2_{args.step2_iter}_feat_{args.max_feat_select}_epoch{args.epochs}_{args.aspect}_{args.embedding}_seed{args.seed}')
else:
    experiment=None

if dist_rank==0:
    print("==========================")


# set ASL loss function
criterion = aslloss.AsymmetricLossOptimized(
        gamma_neg=2, gamma_pos=0,
        clip=0.0,
        disable_torch_grad_focal_loss=False,
        eps=1e-5,
    ).to(local_rank)


# avoid duplicated training
if args.mode == 'train':
    checkpt_file = f'{args.modeldir}/{args.org}_iter1_{args.step1_iter}_iter2_{args.step2_iter}_feat_{args.max_feat_select}_epoch{args.epochs}_{args.aspect}_{args.embedding}_seed{args.seed}.pt'
    if not args.overwrite and os.path.exists(checkpt_file):
        sys.exit(0)


def train_step(model,optimizer,labels,list_mat,list_ind):
    model.train()
    optimizer.zero_grad()
    output = model(list_mat, layer_norm,list_ind)
    outs = nn.functional.sigmoid(output.detach()).cpu().numpy()
    #do not report Fmax during training for acceleration
    if args.noeval_in_train:
        perf_train={'Fmax':0}
    else:
        perf_train = evaluate_performance(labels, outs, (outs > 0.5).astype(int))
    loss_train = criterion(torch.zeros(2,2,2), output, torch.tensor(labels).to(local_rank))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),perf_train['Fmax']


def validate_step(model,labels,list_mat,list_ind):
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm,list_ind)
        outs = nn.functional.sigmoid(output.detach()).cpu().numpy()
        loss_val = criterion(torch.zeros(2,2,2), output, torch.tensor(labels).to(local_rank))
        if args.noeval_in_train:
            perf_val = {'Fmax':0, 'tmax':0}
        else:
            perf_val = evaluate_performance(labels, outs, (outs > 0.5).astype(int))
        return loss_val.item(),perf_val['Fmax'],perf_val['tmax']


def test_step(model,labels,list_mat,list_ind,threshold):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm,list_ind)
        outs = nn.functional.sigmoid(output.detach()).cpu().numpy()
        np.save(f'{args.resultdir}/{args.aspect}_iter1_{args.step1_iter}_iter2_{args.step2_iter}_feat_{args.max_feat_select}_seed{args.seed}-predict.npy', outs)
        loss_test = criterion(torch.zeros(2,2,2), output, torch.tensor(labels).to(local_rank))
        perf_test = evaluate_performance(labels, outs, (outs > threshold).astype(int))
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


def new_optimal_mask(model, model_sel, optimizer_sel, valid_loader, device, num_layer):

    #Calculate input gradients
    equal_masks = torch.ones(num_layer).float().cuda()
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
        
        loss_val_list = []
        for i, (list_val_mat,labels) in enumerate(valid_loader):
            input_val_mat = [list_val_mat[ww] for ww in get_ind]
            for j in range(len(input_val_mat)):
                input_val_mat[j] = input_val_mat[j].to(device)
            loss_val,acc_val,_ = validate_step(model,labels.numpy(),input_val_mat,get_ind)
            loss_val_list.append(loss_val)

        loss_val = np.mean(loss_val_list)
        best_mask_loss.append(loss_val)


    #Find indices with minimum validation loss
    min_loss_ind = np.argmin(best_mask_loss)
    optimal_mask = best_mask[min_loss_ind]


    return optimal_mask, model_sel, model


def train(train_loader,valid_loder,list_test_mat,test_label,num_nodes,num_feat,num_labels):

    num_adj_mat = num_adj
    num_feat_mat = len(list_test_mat) - num_adj

    num_layer = len(list_test_mat)
    model = Classifier_cafa(nfeat=num_feat,
                num_adj_mat=num_adj_mat,
                num_feat_mat=num_feat_mat,
                nhidden=args.hidden,
                nclass=num_labels,
                dropout1=args.dropout1,
                dropout2=args.dropout2,
                dropout3=args.dropout3,
                num_nodes=num_nodes,device=local_rank).to(local_rank)
    model_sel = Selector(num_layer,256).to(local_rank)
    optimizer_select = [
        {'params':model_sel.fc1.parameters(), 'weight_decay':args.wd_sel, 'lr':args.lr_sel},
        {'params':model_sel.fc2.parameters(), 'weight_decay':args.wd_sel, 'lr':args.lr_sel}
    ]

    if parallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                broadcast_buffers=False, find_unused_parameters=True)
        optimizer_sett_classifier = [
            {'params': model.module.fc2.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc2},
            {'params': model.module.fc3.parameters(), 'weight_decay': args.w_fc3, 'lr': args.lr_fc2},
            {'params': model.module.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc1},
        ]
    else:
        optimizer_sett_classifier = [
            {'params': model.fc2.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc2},
            {'params': model.fc3.parameters(), 'weight_decay': args.w_fc3, 'lr': args.lr_fc2},
            {'params': model.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc1},
        ]
    

    optimizer = optim.Adam(optimizer_sett_classifier)
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
    if dist_rank==0:
        print('Step1: Exploration===============')
    for epoch in tqdm(range(args.step1_iter)):
        #choose one subset randomly
        rand_ind = random.choice(combinations)
        #create input to model
        acc_tra_list = []
        loss_tra_list = []
        if parallel:
            train_loader.sampler.set_epoch(epoch)
        for i, (list_train_mat,labels) in enumerate(train_loader):
            input_train_mat = [list_train_mat[ww] for ww in rand_ind]
            for j in range(len(input_train_mat)):
                input_train_mat[j] = input_train_mat[j].to(local_rank)
            loss_tra,acc_tra = train_step(model,optimizer,labels.numpy(),input_train_mat,rand_ind)
            acc_tra_list.append(acc_tra)
            loss_tra_list.append(loss_tra)
        
        acc_val_list = []
        loss_val_list = []
        if parallel:
            valid_loader.sampler.set_epoch(epoch)
        for i, (list_val_mat,labels) in enumerate(valid_loader):
            input_val_mat = [list_val_mat[ww] for ww in rand_ind]
            for j in range(len(input_val_mat)):
                input_val_mat[j] = input_val_mat[j].to(local_rank)
            loss_val,acc_val,_ = validate_step(model,labels.numpy(),input_val_mat,rand_ind)
            acc_val_list.append(acc_val)
            loss_val_list.append(loss_val)

        loss_tra = np.mean(loss_tra_list)
        loss_val = np.mean(loss_val_list)
        acc_tra = np.mean(acc_tra_list)
        acc_val = np.mean(acc_val_list)
            
        #Input mask vector to selector
        input_mask = torch.zeros(num_layer).float().to(local_rank)
        input_mask[list(rand_ind)] = 1.0
        input_loss = torch.FloatTensor([loss_tra]).to(local_rank)
        eval_loss = torch.FloatTensor([loss_val]).to(local_rank)
        loss_select, input_grad = selector_step(model_sel,optimizer_sel,input_mask,eval_loss)
        #loss_select_val = selector_eval(model_sel,input_mask,eval_loss)

        #log metrics
        if args.comet and dist_rank==0:
            experiment.log_metric('train_loss', loss_tra, epoch=epoch+1)
            experiment.log_metric('train_Fmax', acc_tra, epoch=epoch+1)
            experiment.log_metric('valid_loss', loss_val, epoch=epoch+1)
            experiment.log_metric('valid_Fmax', acc_val, epoch=epoch+1)


    #Starting Step-2: Exploitation
    if dist_rank==0:
        print('Step2: Exploitation===============')
    dict_check_loss = dict()
    for epoch in tqdm(range(args.epochs)):

        if epoch<sec_iter:
            #Upto sec_iter epoches optimal subsets are identified
            train_mask, model_sel, model = new_optimal_mask(model, model_sel, optimizer_sel, valid_loader, device, num_layer)

        if epoch==sec_iter:
            min_ind = min(list(dict_check_loss.keys()))
            train_mask = dict_check_loss[min_ind]

        acc_tra_list = []
        loss_tra_list = []
        if parallel:
            train_loader.sampler.set_epoch(epoch)
        for i, (list_train_mat,labels) in enumerate(train_loader):
            input_train_mat = [list_train_mat[ww] for ww in train_mask]
            for i in range(len(input_train_mat)):
                input_train_mat[i] = input_train_mat[i].to(local_rank)
            loss_tra,acc_tra = train_step(model,optimizer,labels.numpy(),input_train_mat,train_mask)
            acc_tra_list.append(acc_tra)
            loss_tra_list.append(loss_tra)

        acc_val_list = []
        loss_val_list = []
        if parallel:
            valid_loader.sampler.set_epoch(epoch)
        for i, (list_val_mat,labels) in enumerate(valid_loader):
            input_val_mat = [list_val_mat[ww] for ww in train_mask]
            for j in range(len(input_val_mat)):
                input_val_mat[j] = input_val_mat[j].to(local_rank)
            loss_val,acc_val,tmax = validate_step(model,labels.numpy(),input_val_mat,train_mask)
            acc_val_list.append(acc_val)
            loss_val_list.append(loss_val)

        loss_tra = np.mean(loss_tra_list)
        loss_val = np.mean(loss_val_list)
        acc_tra = np.mean(acc_tra_list)
        acc_val = np.mean(acc_val_list)

        if args.comet and dist_rank==0:
            experiment.log_metric('train_loss', loss_tra, epoch=epoch+1+args.step1_iter)
            experiment.log_metric('train_Fmax', acc_tra, epoch=epoch+1+args.step1_iter)
            experiment.log_metric('valid_loss', loss_val, epoch=epoch+1+args.step1_iter)
            experiment.log_metric('valid_Fmax', acc_val, epoch=epoch+1+args.step1_iter)


        dict_check_loss[loss_val] = train_mask

        if epoch < sec_iter:
            input_mask = torch.zeros(num_layer).float().to(local_rank)
            input_mask[list(train_mask)] = 1.0
            input_loss = torch.FloatTensor([loss_tra]).to(local_rank)
            eval_loss = torch.FloatTensor([loss_val]).to(local_rank)
            loss_select, _ = selector_step(model_sel,optimizer_sel,input_mask,eval_loss)
            #loss_select_val = selector_eval(model_sel,input_mask,eval_loss)


        if epoch>= sec_iter:
            if loss_val < best:
                best = loss_val
                #save model
                if dist_rank==0:
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
    test_out = test_step(model,test_label,input_test_mat,select_ind,tmax)
    perf = test_out[1]
    
    if args.comet and dist_rank==0:
        experiment.log_metric('test_Fmax', perf['Fmax'], step=1)
        experiment.log_metric('test_acc', perf['acc'], step=1)
        experiment.log_metric('test_F1', perf['F1'], step=1)
        experiment.log_metric('test_m-aupr', perf['m-aupr'], step=1)
        experiment.log_metric('test_M-aupr', perf['M-aupr'], step=1)

    return perf, select_ind


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
    model.load_state_dict(torch.load(checkpt_model))
    model.eval()
    with torch.no_grad():
        output = model(input_test_mat, layer_norm, best_mask)
        outs = nn.functional.sigmoid(output.detach()).cpu().numpy()

    return outs



## main process =====================================================
t_total = time.time()
acc_list = []

aspect_dict = {'P':'bp', 'F':'mf', 'C':'cc'}
n_class_dict = {'P':3992, 'F':677, 'C':551}
# mask indices refer to 0:neighborhood, 1:fusion, 2:cooccurence, 3:coexpression, 
# 4:experimental, 5:database, 6:textmining, 7:Esm
best_mask_dict = {'P':[6,7], 'F':[5,7], 'C':[2,6]}
model_dict = {'P':'data/cafa3/all_iter1_500_iter2_10_feat_2_epoch1500_P_AE_seed42.pt',
              'F':'data/cafa3/all_iter1_300_iter2_70_feat_2_epoch1500_F_AE_seed42.pt',
              'C':'data/cafa3/all_iter1_400_iter2_10_feat_2_epoch1500_C_AE_seed42.pt'}


if args.mode == 'predict' and args.checkpt:
    if not os.path.isfile(args.checkpt):
        raise ValueError("Model not found!")
    else:
        temp_args = os.path.split(args.checkpt)[-1].split('_')
        args.step1_iter = int(temp_args[2])
        args.step2_iter = int(temp_args[4])
        args.max_feat_select = int(temp_args[6])
        assert args.aspect == temp_args[8], "Incosistent aspect argument with provided model!"
        args.embedding = temp_args[9]
        model_dict[args.aspect] = args.checkpt
        if args.mask:
            best_mask_dict[args.aspect] = ast.literal_eval(str(args.mask))
        elif args.resultfile:
            res = pd.read_csv(args.resultfile, header=None)
            df = res[(res[12]==args.step1_iter) & (res[13]==args.step2_iter) & (res[15]==args.max_feat_select) & (res[18]==args.aspect) & ((res[19]==args.embedding))]
            if len(df) == 0:
                raise ValueError('Model result not found!')
            else:
                best_mask_dict[args.aspect] = ast.literal_eval(df[26].values[0])
        else:
            raise ValueError("Please provide --mask or --resultfile")


num_labels = n_class_dict[args.aspect]
best_mask = best_mask_dict[args.aspect]
checkpt_model = model_dict[args.aspect]
aspect = aspect_dict[args.aspect]

if args.mode == 'train':
    df_train = pd.read_csv(f'data/cafa3/{aspect}_go_train_index.csv')
    df_valid = pd.read_csv(f'data/cafa3/{aspect}_go_valid_index.csv')
    df_test = pd.read_csv(f'data/cafa3/{aspect}-test.csv')

    labels_train = df_train.iloc[:,2:num_labels+2].to_numpy()
    labels_valid = df_valid.iloc[:,2:num_labels+2].to_numpy()
    labels_test = df_test.iloc[:,2:].to_numpy() 

# blast to create test indices
if args.fasta and not args.txt:
    cmd_db = f'makeblastdb -in data/cafa3/all_cafa_string.fa -dbtype prot -out data/cafa3/cafa3_string_prot_set'
    if os.path.isfile('data/cafa3/cafa3_string_prot_set.phr') and os.path.isfile('data/cafa3/cafa3_string_prot_set.pin') and os.path.isfile(f'data/cafa3/cafa3_string_prot_set.psq'):
        pass
    else:
        print('Building protein library for blastp ... ')
        os.system(cmd_db)
    
    # to ensure 100% output, we do not implement a E-value threshold here, so false positives allowed
    cmd_blast = f'blastp -db data/cafa3/cafa3_string_prot_set -query {args.fasta} -out {args.resultdir}/{aspect}_query_results.txt -outfmt 6 -max_target_seqs 1'
    print('Querying protein library using blastp ... ')
    os.system(cmd_blast)


# parse blastp results
print('Parsing query results ... ')
if args.txt:
    df = pd.read_csv(args.txt,sep='\t',header=None)
else:
    df = pd.read_csv(f'{args.resultdir}/{aspect}_query_results.txt',sep='\t',header=None)
idx = ~df.duplicated(subset=0)
query_id = df.iloc[:,0][idx].values
target_id = df.iloc[:,1][idx].values


# retrieve test indices in all proteins string id list
string_df = pd.read_csv('data/cafa3/all_proteins_id.csv')
taxo_list, off_set = np.unique(string_df['taxo'].values, return_index=True)
off_set_dict = {}
for A,B in zip(taxo_list,off_set):
    off_set_dict[A] = B
string_df['offset'] = [off_set_dict[i] for i in string_df['taxo'].values]
test_idx = string_df.reset_index().groupby('string')['index'].first()[target_id].values
test_taxo = string_df['taxo'].values[test_idx]
test_idx -= string_df['offset'].values[test_idx]


# read preprocessed train/valid data if in 'train' mode 
print('Loading feature matrices ... ')
if args.mode == 'train':
    list_train_mat = []
    list_val_mat = []
    for e in ['neighborhood', 'fusion', 'cooccurence', 'coexpression', 'experimental', 'database', 'textmining']:
        fn = f'data/cafa3/{aspect}_train_all_net_{e}_{args.embedding}.npy'
        fv = f'data/cafa3/{aspect}_valid_all_net_{e}_{args.embedding}.npy'
        y = np.load(fn)
        list_train_mat.append(y)
        y = np.load(fv)
        list_val_mat.append(y)
    del y

    # load subloc+pfam features
    Z = np.load(f'data/cafa3/{aspect}_train_all_Esm2.npy')
    list_train_mat.append(Z)
    Z = np.load(f'data/cafa3/{aspect}_valid_all_Esm2.npy')
    list_val_mat.append(Z)
    del Z


list_test_mat = []
# it would be easier to have several big matrices, but the memory on disk would not reduce
for e in ['neighborhood', 'fusion', 'cooccurence', 'coexpression', 'experimental', 'database', 'textmining']:
    fo = f'{args.resultdir}/{aspect}_test_net_{e}_{args.embedding}.npy'
    if os.path.exists(fo):
        net = np.load(fo)
    else:
        net = np.empty((0,512))
        for i in range(len(test_idx)):
            taxo = test_taxo[i]
            idx = test_idx[i]
            fn = f'data/cafa3/{taxo}/{taxo}_net_{e}_AE.npy'
            y = np.load(fn)
            net = np.vstack((net,y[idx,:]))
        np.save(fo,net)
    list_test_mat.append(net)

fo = f'{args.resultdir}/{aspect}_test_Esm2.npy'
if os.path.exists(fo):
    esm = np.load(fo)
else:
    esm = np.empty((0,1280))
    for i in range(len(test_idx)):
        taxo = test_taxo[i]
        idx = test_idx[i]
        fn = f'data/cafa3/{taxo}/{taxo}_Esm2.npy'
        Z = np.load(fn)
        esm = np.vstack((esm,Z[idx,:]))
    np.save(fo, esm)

list_test_mat.append(esm)
list_test_mat = [torch.from_numpy(mat).float() for mat in list_test_mat]

num_nodes = list_test_mat[-2].shape[1]
num_feat = list_test_mat[-1].shape[1] # feature dim


# in 'train' mode, only the final score matrix (.npy object) will be saved, but no output for score table (.txt)
if args.mode == 'train':
    list_total_acc = []
    print('Start training ... ')
    for i in range(1):
        list_train_mat = [torch.from_numpy(mat).float() for mat in list_train_mat]
        list_val_mat = [torch.from_numpy(mat).float() for mat in list_val_mat]

        train_dataset = multimodesDataset(8, list_train_mat, labels_train)
        valid_dataset = multimodesDataset(8, list_val_mat, labels_valid)
        test_dataset = multimodesDataset(8, list_test_mat, labels_test)
        
        if parallel:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch, shuffle=False,
                num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
        if parallel:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        else:
            valid_sampler = None
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=valid_sampler)
        if parallel:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) #not used
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=test_sampler) #not used


        accuracy_data, best_mask = train(train_loader,valid_loader,list_test_mat,labels_test,num_nodes,num_feat,num_labels)
        num_layer = len(list_train_mat)

        acc_list.append(accuracy_data['Fmax'])
        list_total_acc.append(100-accuracy_data['Fmax'])


    training_time = time.time() - t_total
    if args.comet and dist_rank==0:
        experiment.log_metric('training_time', "{:.4f}s".format(training_time), step=1)

    if dist_rank==0:
        print("Train time: {:.4f}s".format(training_time))
        print(f"Test accuracy: {np.mean(acc_list):.2f}, std: {np.round(np.std(acc_list),2)}")

        fn = f'{args.resultdir}/{args.out}'
        with open(fn, 'a') as f:
            csv.writer(f).writerow([args.w_fc1, args.w_fc2, args.w_fc3, args.dropout1, args.dropout2, args.dropout3, args.lr_fc1, args.lr_fc2, args.lr_sel, args.wd_sel, args.hidden, args.layer_norm, args.step1_iter, args.step2_iter, args.epochs, args.max_feat_select, args.num_adj, args.org, args.aspect, args.embedding, accuracy_data['Fmax'], accuracy_data['F1'], accuracy_data['M-F1'], accuracy_data['acc'], accuracy_data['m-aupr'], accuracy_data['M-aupr'], best_mask])


# in 'predict' mode for prediction only
if args.mode == 'predict':
    print('Generating scores ... ')
    scores = predict(list_test_mat,num_nodes,num_feat,num_labels)
    np.save(f'{args.resultdir}/{aspect}_DualNetGO_scores.npy', scores)
    GO_term = pd.read_csv(f'data/cafa3/{aspect}-test.csv').columns[2:num_labels+2]
    df_result = pd.DataFrame(scores,columns=GO_term)
    df_result['id'] = query_id
    dfm = df_result.melt(id_vars='id',value_vars=GO_term,ignore_index=True)
    dfm.to_csv(f'{args.resultdir}/{aspect}_DualNetGO_scores.txt', index=False, sep='\t', header=None)
