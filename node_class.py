from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
#from process import *
from utils import *
from model import *
import uuid
import pickle
import copy
import itertools
from collections import defaultdict
import torch_sparse
import warnings
from tqdm import tqdm
#from orion.client import report_objective
warnings.filterwarnings("ignore") #temporary ignoring warning from torch_sparse
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.') #Default seed same as GCNII
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--layer', type=int, default=3, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=16, help='hidden dimensions.')
parser.add_argument('--dropout1', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout2', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout3', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='fb100', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--layer_norm',type=int, default=1, help='layer norm')
parser.add_argument('--w_fc3',type=float, default=0.000, help='Weight decay layer-2')
parser.add_argument('--w_fc2',type=float, default=0.000, help='Weight decay layer-2')
parser.add_argument('--w_fc1',type=float, default=0.000, help='Weight decay layer-1')
parser.add_argument('--lr_fc1',type=float, default=0.001, help='Learning rate 2 fully connected layers')
parser.add_argument('--lr_fc2',type=float, default=0.001, help='Learning rate 2 fully connected layers')
parser.add_argument('--lr_sel',type=float, default=0.001, help='Learning rate for selector')
parser.add_argument('--wd_sel',type=float,default=1e-05,help='weight decay selector layer')
parser.add_argument('--step1_iter',type=int, default=400, help='Step-1 iterations')
parser.add_argument('--step2_iter',type=int, default=20, help='Step-2 iterations')
parser.add_argument('--directed',type=int, default=0, help='Undirected:0, Directed:1')
parser.add_argument('--max_feat_select',type=int, default=5, help='Maximum feature matrices that can be selected.')
parser.add_argument('--num_adj',type=int, default=2, help='Number of sparse adjacency matrices(including powers) as input')


args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


#maximum length of subset to find
feat_select = int(args.max_feat_select)
#feat_select = 6
sec_iter = args.step2_iter

layer_norm = bool(int(args.layer_norm))
print("==========================")
print(f"Dataset: {args.data}")
#print(f"Dropout1:{args.dropout1}, Dropout2:{args.dropout2}, Dropout3:{args.dropout3}, layer_norm: {layer_norm}")
#print(f" w_fc2:{args.w_fc2}, w_fc1:{args.w_fc1}, w_sel:{args.wd_sel}, lr_fc1:{args.lr_fc1}, lr_fc2:{args.lr_fc2},lr_sel:{args.lr_sel}, 1st step iter: {args.step1_iter}, 2nd step iter: {args.step2_iter}")
if args.data  == "genius":
    accuracy = eval_rocauc

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

#set number of adjacency matrices in the input data
is_directed = bool(args.directed)
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
    acc_train = accuracy(output, labels)
    loss_train = F.nll_loss(output, labels.to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate_step(model,labels,list_mat,list_ind):
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm,list_ind)
        loss_val = F.nll_loss(output, labels.to(device))
        acc_val = accuracy(output, labels)
        return loss_val.item(),acc_val.item()


def test_step(model,labels,list_mat,list_ind):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm,list_ind)
        loss_test = F.nll_loss(output, labels.to(device))
        acc_test = accuracy(output, labels)
        #print(mask_val)
        return loss_test.item(),acc_test.item()



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

        loss_val,acc_val = validate_step(model,labels,input_val_mat,get_ind)
        best_mask_loss.append(loss_val)


    #Find indices with minimum validation loss
    min_loss_ind = np.argmin(best_mask_loss)
    optimal_mask = best_mask[min_loss_ind]


    return optimal_mask, model_sel, model



def train(list_train_mat,list_val_mat,list_test_mat,list_label,num_nodes,num_feat,num_labels):

    #Can comment following two lines to save memory on GPU
    #though the training will be slower 
    # Commented for arxiv-year and wiki in our case to avoid GPU OOM error in our case
    
    list_train_mat = [mat.to(device) for mat in list_train_mat]
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


    optimizer_sett_classifier = [
        {'params': model.fc2.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc2},
        {'params': model.fc3.parameters(), 'weight_decay': args.w_fc3, 'lr': args.lr_fc2},
        {'params': model.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc1},
    ]

    optimizer = optim.Adam(optimizer_sett_classifier)

    #model_sel = Selector(num_layer,args.hidden).to(device)
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
        loss_val,acc_val = validate_step(model,list_label[1],input_val_mat,rand_ind)

        #Input mask vector to selector
        input_mask = torch.zeros(num_layer).float().to(device)
        input_mask[list(rand_ind)] = 1.0
        input_loss = torch.FloatTensor([loss_tra]).to(device)
        eval_loss = torch.FloatTensor([loss_val]).to(device)
        loss_select, input_grad = selector_step(model_sel,optimizer_sel,input_mask,input_loss)
        #loss_select_val = selector_eval(model_sel,input_mask,eval_loss)


    #Starting Step-2: Exploitation
    print('Step2: Exploitation===============')
    dict_check_loss = dict()
    for epoch in tqdm(range(args.epochs)):

        if epoch<sec_iter:
            #Upto sec_iter epoches optimal subsets are identified
            train_mask, model_sel, model = new_optimal_mask(model, model_sel, optimizer_sel, list_val_mat,device, list_label[1],num_layer)


        if epoch==sec_iter:
            min_ind = min(list(dict_check_loss.keys()))
            train_mask = dict_check_loss[min_ind]


        input_train_mat = [list_train_mat[ww] for ww in train_mask]
        input_val_mat = [list_val_mat[ww] for ww in train_mask]
        loss_tra,acc_tra = train_step(model,optimizer,list_label[0],input_train_mat,train_mask)
        loss_val,acc_val = validate_step(model,list_label[1],input_val_mat,train_mask)


        dict_check_loss[loss_val] = train_mask

        if epoch < sec_iter:
            input_mask = torch.zeros(num_layer).float().to(device)
            input_mask[list(train_mask)] = 1.0
            input_loss = torch.FloatTensor([loss_tra]).to(device)
            eval_loss = torch.FloatTensor([loss_val]).to(device)
            loss_select, _ = selector_step(model_sel,optimizer_sel,input_mask,input_loss)
            #loss_select_val = selector_eval(model_sel,input_mask,eval_loss)

        '''
        if(epoch+1)%1 == 0:

            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))

        '''


        if loss_val < best and epoch>= sec_iter:
            best = loss_val
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

    test_out = test_step(model,list_label[2],input_test_mat,select_ind)
    acc = test_out[1]


    return acc*100

t_total = time.time()
acc_list = []
datastr = args.data
data_loc = "./processed_data/"+datastr+".pickle"
with open(data_loc,"rb") as fopen:
    main_data = pickle.load(fopen)

list_mat, labels, split_idx_list, num_nodes, num_feat, num_labels = main_data
del main_data



list_total_acc = []
for i in range(1):

    #Create training and testing split
    list_train_mat  = []
    list_val_mat = []
    list_test_mat = []
    train_idx = split_idx_list[i]['train']
    valid_idx = split_idx_list[i]['valid']
    test_idx = split_idx_list[i]['test']

    for mat in list_mat[:num_adj]:

        tmp_mat = mat.to_scipy().tocsr()
        tmp_train_mat = scipy_to_tensor(tmp_mat[train_idx,:])
        tmp_valid_mat = scipy_to_tensor(tmp_mat[valid_idx,:])
        tmp_test_mat = scipy_to_tensor(tmp_mat[test_idx,:])

        list_train_mat.append(tmp_train_mat)
        list_val_mat.append(tmp_valid_mat)
        list_test_mat.append(tmp_test_mat)


    for mat in list_mat[num_adj:]:
        if is_directed:
            mat = F.normalize(mat,p=0,dim=1)
        if args.data == "snap-patents":
            #sparsify snap-patents features to reduce memory
            list_train_mat.append(mat[train_idx,:].to_sparse())
            list_val_mat.append(mat[valid_idx,:].to_sparse())
            list_test_mat.append(mat[test_idx,:].to_sparse())
        else:
            list_train_mat.append(mat[train_idx,:])
            list_val_mat.append(mat[valid_idx,:])
            list_test_mat.append(mat[test_idx,:])

    list_label = [labels[train_idx].reshape(-1), labels[valid_idx].reshape(-1), labels[test_idx].reshape(-1)]

    del mat
    del tmp_train_mat
    del tmp_valid_mat
    del tmp_test_mat



    accuracy_data = train(list_train_mat,list_val_mat,list_test_mat,list_label,num_nodes,num_feat,num_labels)
    num_layer = len(list_train_mat)

    acc_list.append(accuracy_data)
    list_total_acc.append(100-accuracy_data)



print("Train time: {:.4f}s".format(time.time() - t_total))
print(f"Test accuracy: {np.mean(acc_list):.2f}, {np.round(np.std(acc_list),2)}")


