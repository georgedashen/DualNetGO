import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Classifier(nn.Module):
    def __init__(self,nfeat,num_adj_mat,num_feat_mat,nhidden,nclass,dropout1,dropout2, dropout3, num_nodes, device=0):
        super(Classifier,self).__init__()
        self.fc1 = nn.ModuleList()
        for _ in range(num_adj_mat):
            self.fc1.append(nn.Sequential(nn.Linear(num_nodes,nhidden)))

        for _ in range(num_feat_mat):
            self.fc1.append(nn.Linear(nfeat,nhidden))
        self.fc2 = nn.Linear(nhidden,nhidden)
        self.fc3 = nn.Linear(nhidden,nclass)
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.act_fn = nn.ReLU()
        self.device = device

    
    def forward(self,list_mat,layer_norm,list_ind):

        list_out = list()
        device = self.device
        #Select matrices
        for ind, ind_m in enumerate(list_ind):
            tmp_out = self.fc1[ind_m](list_mat[ind].cuda(device))
            if layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1)
            tmp_out = self.act_fn(tmp_out)
            tmp_out = F.dropout(tmp_out,self.dropout1,training=self.training)
            list_out.append(tmp_out)

        final_mat = torch.zeros_like(list_out[0]).cuda(device)
        for mat in list_out:
            final_mat += mat
        final_mat = final_mat/len(list_mat)

        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout2,training=self.training)
        out = self.fc2(out)
        out = F.normalize(out,p=2,dim=1)
        out = self.act_fn(out)
        out = F.dropout(out,self.dropout3,training=self.training)
        out = self.fc3(out)

        return F.log_softmax(out, dim=1)


class Classifier_evidence(nn.Module):
    def __init__(self,nfeat,nhidden,nclass,dropout1,dropout2,dropout3,num_nodes,device=0):
        super(Classifier_evidence,self).__init__()
        self.fc1 = nn.ModuleList()
        for _ in range(4):
            self.fc1.append(nn.Sequential(nn.Linear(512,nhidden)))

        for _ in range(1):
            self.fc1.append(nn.Sequential(nn.Linear(num_nodes,nhidden)))

        for _ in range(1):
            self.fc1.append(nn.Linear(nfeat,nhidden))

        self.fc2 = nn.Linear(nhidden,nhidden)
        self.fc3 = nn.Linear(nhidden,nclass)
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.act_fn = nn.ReLU()
        self.device = device

    
    def forward(self,list_mat,layer_norm,list_ind):

        list_out = list()
        device = self.device
        #Select matrices
        for ind, ind_m in enumerate(list_ind):
            tmp_out = self.fc1[ind_m](list_mat[ind].cuda(device))
            if layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1)
            tmp_out = self.act_fn(tmp_out)
            tmp_out = F.dropout(tmp_out,self.dropout1,training=self.training)
            list_out.append(tmp_out)

        final_mat = torch.zeros_like(list_out[0]).cuda(device)
        for mat in list_out:
            final_mat += mat
        final_mat = final_mat/len(list_mat)

        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout2,training=self.training)
        out = self.fc2(out)
        out = F.normalize(out,p=2,dim=1)
        out = self.act_fn(out)
        
        return F.log_softmax(out, dim=1)


class Selector(nn.Module):
    def __init__(self,mask_size,nhidden):
        super (Selector,self).__init__()
        self.fc1 = nn.Linear(mask_size,nhidden)
        self.fc2 = nn.Linear(nhidden,1)
        self.act_fn = nn.ReLU()

    def forward(self,mask,loss):
        out = self.fc1(mask)
        out = self.act_fn(out)
        out = F.dropout(out,0.5,training=self.training)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    pass







