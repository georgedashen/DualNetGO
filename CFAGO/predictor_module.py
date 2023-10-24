# --------------------------------------------------------
# part of codes borrowed from Quert2Label
# Written by Zhourun wu
# --------------------------------------------------------

import os, sys
import os.path as osp

import torch
from torch import Tensor
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math
import copy

from multihead_attention import _get_activation_fn

class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x
    
    

class Qeruy2Label_Decoder(nn.Module):
    def __init__(self, DecoderTransformer, num_class):
        super().__init__()
        self.DecoderTransformer = DecoderTransformer
        self.num_class = num_class

        hidden_dim = DecoderTransformer.d_model
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

    def forward(self, x):
        
        query_input = self.query_embed.weight
        pos = None
        hs = self.DecoderTransformer(x, query_input, pos)[0] # B,K,d
        out = self.fc(hs[-1])
        
        # import ipdb; ipdb.set_trace()
        return out

    def finetune_paras(self):
        from itertools import chain
        return chain(self.DecoderTransformer.parameters(), self.fc.parameters(), self.query_embed.parameters())


class FC_Decoder(nn.Module):
    def __init__(self, num_class, dim_feedforward, activation, dropout):
        super().__init__()
        self.num_class = num_class
        
        self.output_layer1 = nn.Linear(dim_feedforward * 2, dim_feedforward // 2)
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.output_layer3 = nn.Linear(dim_feedforward // 2, num_class)

    def forward(self, hs):
        hs = hs.permute(1,0,2)
        hs = hs.flatten(1)
        
        hs = self.output_layer1(hs)
        hs = self.activation1(hs)
        hs = self.dropout1(hs)

        out = self.output_layer3(hs)
        return out

class Predictor(nn.Module):
    def __init__(self, pre_model, num_class, dim_feedforward, activation, dropout):
        super().__init__()
        self.pre_model = pre_model
        self.fc_decoder = FC_Decoder(
                            num_class = num_class,
                            dim_feedforward = dim_feedforward,
                            activation = activation,
                            dropout = dropout
                         )

    def forward(self, src):
        rec, hs = self.pre_model(src)
        out = self.fc_decoder(hs)
        return rec, out
        
def build_predictor(pre_model, args):
    predictor = Predictor(
        pre_model = pre_model,
        num_class = args.num_class,
        dim_feedforward = args.dim_feedforward,
        activation = _get_activation_fn(args.activation),
        dropout = args.dropout
    )

    return predictor