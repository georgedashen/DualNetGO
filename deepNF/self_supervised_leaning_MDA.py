import argparse
import math
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from get_dataset import get_ssl_datasets

from logger import setup_logger
import loss
from model import build_MDA_encoder
from torch.nn.functional import normalize

import copy
import csv

def parser_args():
    parser = argparse.ArgumentParser(description='MDA self-supervised Training')
    parser.add_argument('--org', type=str, default='human', help='organism')
    parser.add_argument('--dataset_dir', help='dir of dataset', default='../data')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--activation', type=str, default='sigmoid')

    # loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')            
    parser.add_argument('--loss_clip', default=0.0, type=float,
                                            help='scale factor for clip')  
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs')
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the multi-head attention module")

    args = parser.parse_args()
    return args

def get_args():
    args = parser_args()
    return args


def main():
    args = get_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
    
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output)
    
    return main_worker(args, logger)

def main_worker(args, logger):
    full_dataset, args.modesfeature_len = get_ssl_datasets(args)
    
    # build model
    pre_model = build_MDA_encoder(args, args.modesfeature_len)
    pre_model = pre_model.cuda()

    # criterion
    pre_criterion = loss.pretrainLossOptimized_MDA(
        clip=args.loss_clip,
        eps=args.eps,
    )

    # optimizer
    #if args.optim == 'AdamW':
    pre_model_param_dicts = [
        {"params": [p for n, p in pre_model.named_parameters() if p.requires_grad]},
    ]
    pre_model_optimizer = getattr(torch.optim, 'AdamW')(
        pre_model_param_dicts,
        args.lr,
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01
    )

    # tensorboard
    
    full_sampler = None
    full_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=args.batch_size, shuffle=(full_sampler is not None),
        num_workers=args.workers, pin_memory=True, sampler=full_sampler, drop_last=False)
    
    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)
    
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, losses_ema],
        prefix='=> Test Epoch: ')

    end = time.time()

    torch.cuda.empty_cache()
    
    pre_loss_f = args.output + '/' + args.org + '_lr_' + str(args.lr) + '_seed_' + str(args.seed) + \
                 '_activation_' + str(args.activation) + '_pre_loss.csv'
    with open(pre_loss_f, 'w') as f:
        csv.writer(f).writerow(['pre_loss'])
    steplr = lr_scheduler.StepLR(pre_model_optimizer, 2500)
    for epoch in range(args.epochs):
        print('epoch=', epoch)
        pre_loss = pre_train(full_loader, pre_model, pre_criterion, pre_model_optimizer, steplr, epoch, args, logger)
            
        print('epoch={}, pre_loss={}'.format(epoch,pre_loss))
        
        with open(pre_loss_f, 'a') as f:
            csv.writer(f).writerow([pre_loss, pre_model_optimizer.param_groups[0]['lr']])
        
    if args.save_model:
        torch.save(pre_model, args.output + '/' + args.org + '_lr_' + str(args.lr) + '_seed_' + str(args.seed) + \
               '_activation_' + str(args.activation) + '_model_MDA.pkl')

    with torch.no_grad():
        pre_model.eval()
        output = []
        for i, proteins in enumerate(full_loader):
            for k in range(len(proteins)):
                proteins[k] = proteins[k].cuda()
            rec, hs = pre_model(proteins)
            output.append(hs)
        
        output = torch.cat(output,dim=0)
        output = output.detach().cpu().numpy()
        print(output.shape)

    np.save(f'{args.output}/{args.org}_net_MDA.npy', output)


def pre_train(full_loader, pre_model, pre_criterion, optimizer, steplr, epoch, args, logger):
    losses = AverageMeter('Loss', ':5.3f')

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    pre_model.train()
    for i, proteins in enumerate(full_loader):
        for k in range(len(proteins)):
            proteins[k] = proteins[k].cuda()
        ori = proteins
        rec, hs = pre_model(proteins)
        loss = pre_criterion(ori, rec, hs)
                
        # record loss
        losses.update(loss.item(), rec[0].shape[0])
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    steplr.step()
    return losses.avg

##################################################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeterHMS(AverageMeter):
    """Meter for timer in HH:MM:SS format"""
    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name, 
                             val=str(datetime.timedelta(seconds=int(self.val))), 
                             sum=str(datetime.timedelta(seconds=int(self.sum))))

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
