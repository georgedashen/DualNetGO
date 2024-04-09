# --------------------------------------------------------
# part of code borrowed from Quert2Label
# Written by Zhourun Wu
# Revised by Zhuoyang CHEN
# --------------------------------------------------------

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
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from get_dataset import get_ssl_datasets

from logger import setup_logger
import aslloss
from encoder import build_MLPAE_Encoder
from validation import evaluate_performance
from torch.nn.functional import normalize

import copy
import csv

def parser_args():
    parser = argparse.ArgumentParser(description='CFAGO self-supervised Training')
    parser.add_argument('--org', help='organism')
    parser.add_argument('--dataset_dir', help='dir of dataset', default='../data')
    parser.add_argument('--aspect', type=str, choices=['P', 'F', 'C'], help='GO aspect')
    parser.add_argument('--evidence', default='combined', choices = ['neighborhood', 'fusion','cooccurence', 'coexpression', 'experimental', 'database', 'textmining', 'combined'], help='what evidence is used to construct the PPI graph')
    parser.add_argument('--output', metavar='DIR', 
                        help='path to output folder')
    parser.add_argument('--num_class', default=45, type=int,
                        help="Number of class labels")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--feature', action="store_true", default=False, help='The encoded matrix is a feature matrix or not')
    parser.add_argument('--embed_len', type=str, default=512)

    # loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')            
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                                            help='scale factor for loss')
    parser.add_argument('--loss_clip', default=0.0, type=float,
                                            help='scale factor for clip')  

    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs')

    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')


    # distribution training
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')


    # data aug
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')              
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ') 

    parser.add_argument('--norm_norm', action='store_true', default=False,
                        help='using mormal scale to normalize input features')

    # * Transformer
    parser.add_argument('--attention_layers', default=6, type=int, 
                        help="Number of layers of each multi-head attention module")
    
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the multi-head attention blocks")
    parser.add_argument('--activation', default='gelu', type=str, choices=['relu', 'gelu', 'lrelu', 'sigmoid'],
                        help="Number of attention heads inside the multi-head attention module's attentions")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the multi-head attention module")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the multi-head attention module's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # training
    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')
    args = parser.parse_args()
    return args

def get_args():
    args = parser_args()
    return args


best_mAP = 0

def main():
    args = get_args()
    
    if 'WORLD_SIZE' in os.environ:
        assert args.world_size > 0, 'please set --world-size and --rank in the command line'
        # launch by torch.distributed.launch
        # Single node
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...
        # Multi nodes
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 0 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 1 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print('os.environ:', os.environ)
    else:
        # single process, useful for debugging
        #   python main.py ...
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
    
    torch.cuda.set_device(args.local_rank)
    print('| distributed init (local_rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)
    cudnn.benchmark = True
    

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Q2L")
    
    args.h_n = 1
    return main_worker(args, logger)

def main_worker(args, logger):
    global best_mAP
    full_dataset, args.modesfeature_len = get_ssl_datasets(args)
    if args.feature:
        feature_len = args.modesfeature_len[1]
    else:
        feature_len = args.modesfeature_len[0]
    args.encode_structure = [1024]
    
    # build model
    pre_model = build_MLPAE_Encoder(feature_len, args)
    pre_model = pre_model.cuda()


    # criterion
    pre_criterion = aslloss.pretrainLossOptimized_MLPAE(
        clip=args.loss_clip,
        eps=args.eps,
    )

    # optimizer
    args.lr_mult = 1
    #if args.optim == 'AdamW':
    pre_model_param_dicts = [
        {"params": [p for n, p in pre_model.named_parameters() if p.requires_grad]},
    ]
    pre_model_optimizer = getattr(torch.optim, 'AdamW')(
        pre_model_param_dicts,
        args.lr_mult * args.lr,
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01
    )

    # tensorboard
    
    full_sampler = None
    full_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=args.batch_size, shuffle=(full_sampler is not None),
        num_workers=args.workers, pin_memory=True, sampler=full_sampler, drop_last=False)
    

    if args.evaluate:
        _, perf = validate(val_loader, model, criterion, args, logger, val_dataset)
        #logger.info(' * perf {m_aupr:.5f, M_aupr:.5f, fmax:.5f, acc:.5f}'
        #      .format(perf['m-aupr'], perf['M-aupr'], perf['F1'], perf['acc']))
        return
    

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)
    
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, losses_ema],
        prefix='=> Test Epoch: ')

    # one cycle learning rate
    #pre_model_scheduler = lr_scheduler.OneCycleLR(pre_model_optimizer, max_lr=args.lr, steps_per_epoch=full_dataset[0].shape[0]//args.batch_size, epochs=args.epochs, pct_start=0.2)
    
    end = time.time()

    torch.cuda.empty_cache()
    
    pre_loss_f = args.output + '/' + args.org + '_attention_layers_' + str(args.attention_layers) + '_lr_' + str(args.lr) + '_seed_' + str(args.seed) + \
                 '_activation_' + str(args.activation) + '_pre_loss.csv'
    with open(pre_loss_f, 'w') as f:
        csv.writer(f).writerow(['pre_loss'])
    steplr = lr_scheduler.StepLR(pre_model_optimizer, 2500)
    for epoch in range(args.start_epoch, args.epochs):
        print('epoch=', epoch)
        pre_loss = pre_train(full_loader, pre_model, pre_criterion, pre_model_optimizer, steplr, epoch, args, logger)
            
        print('epoch={}, pre_loss={}'.format(epoch,pre_loss))
        
        with open(pre_loss_f, 'a') as f:
            csv.writer(f).writerow([pre_loss, pre_model_optimizer.param_groups[0]['lr']])
        
    if args.save_model:
        if args.feature:
            torch.save(pre_model, args.output + '/' + args.org + '_attention_layers_' + str(args.attention_layers) + '_lr_' + str(args.lr) + '_seed_' + str(args.seed) + \
               '_activation_' + str(args.activation) + '_model_feature_MLPAE.pkl')
        else:
            torch.save(pre_model, args.output + '/' + args.org + '_attention_layers_' + str(args.attention_layers) + '_lr_' + str(args.lr) + '_seed_' + str(args.seed) + \
               '_activation_' + str(args.activation) + '_model_' + args.evidence + '_MLPAE.pkl')

    with torch.no_grad():
        pre_model.eval()
        output = []
        for i, proteins in enumerate(full_loader):
            proteins[0] = proteins[0].cuda()
            proteins[1] = proteins[1].cuda()
            if args.feature:
                protein = proteins[1]
            else:
                protein = proteins[0]
            rec, hs = pre_model(protein)
            output.append(hs)
        
        output = torch.cat(output,dim=0)
        output = output.detach().cpu().numpy()

    if args.feature:
        np.save(f'{args.dataset_dir}/{args.org}/{args.org}_net_feature_MLPAE.npy', output)
    else:
        np.save(f'{args.dataset_dir}/{args.org}/{args.org}_net_{args.evidence}_MLPAE.npy', output)

            

def pre_train(full_loader, pre_model, pre_criterion, optimizer, steplr, epoch, args, logger):
    losses = AverageMeter('Loss', ':5.3f')

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    pre_model.train()
    for i, proteins in enumerate(full_loader):
        proteins[0] = proteins[0].cuda() #adj_mat
        proteins[1] = proteins[1].cuda() #domain features
        if args.feature:
            protein = proteins[1]
        else:
            protein = proteins[0]
        ori = protein.clone()
        rec, hs = pre_model(protein)
        loss = pre_criterion(ori, rec)
        if args.loss_dev > 0:
            loss *= args.loss_dev
                
        # record loss
        losses.update(loss.item(), rec.size(0))
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    steplr.step()
    return losses.avg

##################################################################################
def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()

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

class myRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data, generator, shuffle=True):
        self.data = data
        self.generator = generator
        self.shuffle = shuffle
    def __iter__(self):
        n = len(self.data)
        if self.shuffle:
            return iter(torch.randperm(n, generator=self.generator).tolist())
        else:
            return iter(list(range(n)))
    def __len__(self):
        return len(self.data)

def kill_process(filename:str, holdpid:int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True, cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist

if __name__ == '__main__':
    main()
