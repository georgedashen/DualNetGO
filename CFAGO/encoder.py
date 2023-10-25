# --------------------------------------------------------
# part of code borrowed from Quert2Label
# Written by Zhourun Wu
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

from multihead_attention import build_transformerEncoder, _get_activation_fn

class PreEncoder(nn.Module):
    def __init__(self, transformerEncoder, transformerDecoder, feature_len, activation, dropout):
        super().__init__()
        self.transformerEncoder = transformerEncoder
        self.transformerDecoder = transformerDecoder
        dim_feedforward = transformerEncoder.dim_feedforward
        self.input_proj_x1 = nn.Linear(feature_len[0], dim_feedforward * 2)
        self.input_proj_z1 = nn.Linear(feature_len[1], dim_feedforward * 2)
        self.input_proj_x2 = nn.Linear(dim_feedforward * 2, dim_feedforward)
        self.input_proj_z2 = nn.Linear(dim_feedforward * 2, dim_feedforward)
        
        self.norm_x1 = nn.LayerNorm(dim_feedforward * 2)
        self.norm_z1 = nn.LayerNorm(dim_feedforward * 2)
        self.norm_x2 = nn.LayerNorm(dim_feedforward)
        self.norm_z2 = nn.LayerNorm(dim_feedforward)
        
        self.dropout_x1 = nn.Dropout(dropout)
        self.dropout_z1 = nn.Dropout(dropout)
        self.dropout_x2 = nn.Dropout(dropout)
        self.dropout_z2 = nn.Dropout(dropout)
        
        self.activation_x1 = copy.deepcopy(activation)
        self.activation_z1 = copy.deepcopy(activation)
        self.activation_x2 = copy.deepcopy(activation)
        self.activation_z2 = activation
        
        self.W_x1 = nn.Linear(dim_feedforward, dim_feedforward * 2)
        self.W_z1 = nn.Linear(dim_feedforward, dim_feedforward * 2)
        self.W_x2 = nn.Linear(dim_feedforward * 2, feature_len[0])
        self.W_z2 = nn.Linear(dim_feedforward * 2, feature_len[1])
        
        self.activation_wx = copy.deepcopy(activation)
        self.activation_wz = copy.deepcopy(activation)
        
        self.dropout_wx = nn.Dropout(dropout)
        self.dropout_wz = nn.Dropout(dropout)
        
        self.norm_wx = nn.LayerNorm(dim_feedforward * 2)
        self.norm_wz = nn.LayerNorm(dim_feedforward * 2)
        
    def forward(self, src):
        in_x = src[0]
        in_x = self.input_proj_x1(in_x)
        in_x = self.norm_x1(in_x)
        in_x = self.activation_x1(in_x)
        in_x = self.dropout_x1(in_x)
        in_x = in_x
        
        in_z = src[1]
        in_z = self.input_proj_z1(in_z)
        in_z = self.norm_z1(in_z)
        in_z = self.activation_z1(in_z)
        in_z = self.dropout_z1(in_z)
        in_z = in_z
        
        in_x = in_x
        in_x = self.input_proj_x2(in_x)
        in_x = self.norm_x2(in_x)
        in_x = self.activation_x2(in_x)
        in_x = self.dropout_x2(in_x)
        in_x = in_x
        
        in_z = in_z
        in_z = self.input_proj_z2(in_z)
        in_z = self.norm_z2(in_z)
        in_z = self.activation_z2(in_z)
        in_z = self.dropout_z2(in_z)
        in_z = in_z
        
        in_x = in_x.unsqueeze(0)
        in_z = in_z.unsqueeze(0)
        
        in_put = torch.cat([in_x, in_z], 0)
        
        hs = self.transformerEncoder(in_put) # B,K,d
        rec = self.transformerDecoder(hs) # B,K,d
        
        ph_x = self.W_x1(rec[0])
        ph_x = self.norm_wx(ph_x)
        ph_x = self.activation_wx(ph_x)
        ph_x = self.dropout_wx(ph_x)
        
        ph_z = self.W_z1(rec[1])
        ph_z = self.norm_wz(ph_z)
        ph_z = self.activation_wz(ph_z)
        ph_z = self.dropout_wz(ph_z)
        
        rec_x = self.W_x2(ph_x)
        rec_z = self.W_z2(ph_z)
        # import ipdb; ipdb.set_trace()
        
        return (rec_x, rec_z), hs


def build_PreEncoder(args):
    transformerEncoder = build_transformerEncoder(args)
    transformerDecoder = build_transformerEncoder(args)

    model = PreEncoder(
        transformerEncoder = transformerEncoder,
        transformerDecoder = transformerDecoder,
        feature_len = args.modesfeature_len,
        activation = _get_activation_fn(args.activation),
        dropout = args.dropout
    )

    return model
        
        
class MLPAE_Encoder(nn.Module):
    def __init__(self, feature_len, embed_len, activation, dropout):
        super().__init__()
        self.input_proj_1 = nn.Linear(feature_len, embed_len * 2)
        self.input_proj_2 = nn.Linear(embed_len * 2, embed_len)

        self.norm_1 = nn.LayerNorm(embed_len * 2)
        self.norm_2 = nn.LayerNorm(embed_len)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.activation_1 = activation
        self.activation_2 = copy.deepcopy(activation)

        self.W_1 = nn.Linear(embed_len, embed_len * 2)
        self.W_2 = nn.Linear(embed_len * 2, feature_len)

        self.activation_wx = copy.deepcopy(activation)

        self.dropout_wx = nn.Dropout(dropout)
        self.dropout_wz = nn.Dropout(dropout)

        self.norm_wx = nn.LayerNorm(embed_len * 2)
        
    def forward(self, src):
        in_x = src
        in_x = self.input_proj_1(in_x)
        in_x = self.norm_1(in_x)
        in_x = self.activation_1(in_x)
        in_x = self.dropout_1(in_x)

        in_x = self.input_proj_2(in_x)
        in_x = self.norm_2(in_x)
        in_x = self.activation_2(in_x)
        in_x = self.dropout_2(in_x)
        hs = in_x.clone()

        ph_x = self.W_1(hs)
        ph_x = self.norm_wx(ph_x)
        ph_x = self.activation_wx(ph_x)
        ph_x = self.dropout_wx(ph_x)

        rec = self.W_2(ph_x)

        return rec, hs

def build_MLPAE_Encoder(feature_len, args):
    model = MLPAE_Encoder(
        feature_len = feature_len,
        embed_len = args.embed_len,
        activation = _get_activation_fn(args.activation),
        dropout = args.dropout
    )

    return model