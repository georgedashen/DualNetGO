# --------------------------------------------------------
#split for train, valid, test
# part of code borrowed from Quert2Label
# Rewritten by Zhuoyang CHEN
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
from tqdm import tqdm

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

from get_dataset import get_datasets
from encoder import build_PreEncoder
import aslloss
from predictor_module import build_predictor
from validation import evaluate_performance
from torch.nn.functional import normalize
from logger import setup_logger
import copy
import csv

#from comet_ml import Experiment
#from comet_ml.integration.pytorch import log_model

example_usage = 'CUDA_VISIBLE_DEVICES=7 python CFAGO_split_comet.py --org human --dataset_dir ../Dataset/human --output human_result_comet --aspect P --num_class 45 --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.3 --attention_layers 6 --gamma_pos 0 --gamma_neg 2 --batch-size 32 --activation gelu --lr 1e-4 --pretrained_model human_result/human_attention_layers_6_lr_1e-05_seed_1329765522_activation_gelu_model.pkl'


def parser_args():
    parser = argparse.ArgumentParser(description='CFAGO main', epilog=example_usage)
    parser.add_argument('--org', help='organism')
    parser.add_argument('--dataset_dir', help='dir of dataset')
    parser.add_argument('--aspect', default='P', type=str, choices=['P', 'F', 'C'], help='GO aspect')
    parser.add_argument('--pretrained_model', type=str, help='pretrained self-supervide learning model')
    parser.add_argument('--output', metavar='DIR', 
                        help='path to output folder')
    parser.add_argument('--num_class', default=45, type=int,
                        help="Number of class labels")
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')
    parser.add_argument('--EndFreezeEpoch', type=int, default=50, help='Epoch to end AE freeze and train the whole structure')

    # loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=False, 
                        help='disable_torch_grad_focal_loss in asl')            
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
    parser.add_argument('--save_model', type=bool, default=False)

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
    parser.add_argument('--comet', action='store_true', default=False)
    args = parser.parse_args()
    return args

def get_args():
    args = parser_args()
    return args


best_mAP = 0

def set_rand_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    args = get_args()
    torch.autograd.set_detect_anomaly(True)
    
    if args.comet:
        with open('/home/zhuoyang/comet_API_token','r') as f:
            experiment = Experiment(f.read().rstrip(), project_name="CFAGO_split")
            experiment.set_name(f'CFAGO_{args.aspect}')
            experiment.log_parameters(vars(args))
    else:
        experiment = None

    
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
        set_rand_seed(args.seed)
    
    torch.cuda.set_device(args.local_rank)
    print('| distributed init (local_rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)
    cudnn.benchmark = True
    

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="CFAGO")

    args.h_n = 1
    return main_worker(args, logger, experiment)

def main_worker(args, logger, experiment):
    global best_mAP
    train_dataset, valid_dataset, test_dataset, args.modesfeature_len = get_datasets(args)
    criterion = aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        clip=args.loss_clip,
        disable_torch_grad_focal_loss=args.dtgfl,
        eps=args.eps,
    )

    # optimizer
    args.lr_mult = args.batch_size / 32

    # tensorboard

    # optionally resume from a checkpoint    

    # Data loading code
    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_sampler = None
    #assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    
    #val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    
    test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)
    

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
    best_epoch = -1
    best_finetune_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_regular_finetune_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    torch.cuda.empty_cache()
    
    
    fn = args.output + '/' + args.org + '_attention_layers_' + str(args.attention_layers) + '_aspect_' + args.aspect + '_fintune_seed_' + str(args.seed) + \
         '_act_' + args.activation + '.csv'
    with open(fn, 'w') as f:
        csv.writer(f).writerow(['m-aupr','M-aupr','F1','acc', 'Fmax'])
    
    for epoch in range(1):
        if args.seed is not None:
            set_rand_seed(args.seed)
        torch.cuda.empty_cache()
        

        finetune_pre_model = torch.load(args.pretrained_model)
        predictor_model = build_predictor(finetune_pre_model, args)
        predictor_model = predictor_model.cuda()
        
        predictor_model_param_dicts = [
            {"params": [p for n, p in predictor_model.pre_model.named_parameters() if p.requires_grad], "lr":1e-5},
            {"params": [p for n, p in predictor_model.fc_decoder.named_parameters() if p.requires_grad]}
        ]
        
        predictor_model_optimizer = getattr(torch.optim, 'AdamW')(
            predictor_model_param_dicts,
            lr = args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01
        )
        steplr = lr_scheduler.StepLR(predictor_model_optimizer, gamma=0.1, step_size=50)
        patience = 10
        changed_lr = False

        max_fmax = 0
        max_epoch = 0
        
        for epoch_train in tqdm(range(100)):            
            # train for one epoch
            print(f'Epoch {epoch_train} starts:')
            args.current_epoch = epoch_train + 1
            train_loss = train(train_loader, predictor_model, criterion, 
                               predictor_model_optimizer, steplr, epoch_train, args, logger)
            print('Train Loss = ', train_loss)
    
            old_loss = train_loss

            val_loss, perf = evaluate(val_loader, predictor_model, criterion, args, logger, experiment)
            print(f'Validation loss = {val_loss}, acc. = {perf["acc"]}, F1 = {perf["F1"]}, M-F1 = {perf["M-F1"]}, Fmax = {perf["Fmax"]}')

            if args.comet:
                experiment.log_metric('train_loss', train_loss, epoch=epoch_train+1)
                experiment.log_metric('validation_loss', val_loss, epoch=epoch_train+1)
                experiment.log_metric('validation_accuracy', perf["acc"], epoch=epoch_train+1)
                experiment.log_metric('validation_F1', perf["F1"], epoch=epoch_train+1)
                experiment.log_metric('validation_Fmax', perf["Fmax"], epoch=epoch_train+1)

            if perf["Fmax"] > max_fmax:
                max_epoch = epoch_train
                max_fmax = perf["Fmax"]
                threshold = perf['tmax']

                # evaluate on testing set
                test_loss, test_perf = evaluate(test_loader, predictor_model, criterion, args, logger, experiment, threshold, confusion=True)
                print(f'Epoch: {max_epoch+1}, Test loss = {test_loss}, acc. = {test_perf["acc"]}, F1 = {test_perf["F1"]}, M-F1 = {test_perf["M-F1"]}, Fmax = {test_perf["Fmax"]}')
        
                with open(fn, 'a') as f:
                    csv.writer(f).writerow([test_perf['m-aupr'], test_perf['M-aupr'], test_perf['F1'], test_perf['M-F1'], test_perf['acc'], test_perf['Fmax']])
        
        if args.comet:
            experiment.log_metric('test_Fmax', test_perf['Fmax'], step=1)
            experiment.log_metric('test_acc', test_perf['acc'], step=1)
            experiment.log_metric('test_F1', test_perf['F1'], step=1)
        
    #log_model(experiment, predictor_model, model_name="domain/adjMat-AE + MLP")
    if args.save_model:    
        torch.save(predictor_model.state_dict(), args.output+'/model.pth')
    #the_model = TheModelClass(*args, **kwargs)
    #the_model.load_state_dict(torch.load(PATH))

    return 0



########### train =============================
def train(train_loader, predictor_model, criterion, optimizer, steplr, epoch_train, args, logger):
    
    losses = AverageMeter('Loss', ':5.3f')
    #lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    def get_learning_rate(optimizer):
        return optimizer.param_groups[1]["lr"]

    #lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))
    
    predictor_model.train()
    # switch to train mode
    
    if epoch_train >= args.EndFreezeEpoch:
        for p in predictor_model.pre_model.parameters():
            p.requires_grad = True
    else:
        #finetune_pre_model.eval()
        for p in predictor_model.pre_model.parameters():
            p.requires_grad = False
    
    end = time.time()
    for i, (proteins, label) in enumerate(train_loader):
        # measure data loading time
        for i in range(len(proteins)):
            proteins[i] = proteins[i].cuda()
            
        label = label.cuda()
        # compute output
        
        rec, output = predictor_model(proteins)
        loss = criterion(rec, output, label)
        if args.loss_dev > 0:
            loss *= args.loss_dev
                
        # record loss
        losses.update(loss.item(), proteins[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    steplr.step()
    return losses.avg



######### evaluate =======================
@torch.no_grad()
def evaluate(test_loader, predictor_model, criterion, args, logger, experiment, threshold=0.5, confusion=False):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    # Acc1 = AverageMeter('Acc@1', ':5.2f')
    # top5 = AverageMeter('Acc@5', ':5.2f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    # mAP = AverageMeter('mAP', ':5.3f', val_only=)

    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    # switch to evaluate mode
    saveflag = False
    predictor_model.eval()
    saved_data = []
    with torch.no_grad():
        for i, (proteins, label) in enumerate(test_loader):
            for i in range(len(proteins)):
                proteins[i] = proteins[i].cuda()
            label = label.cuda()

            # compute output
            rec, output  = predictor_model(proteins)
            loss = criterion(rec, output, label)
            if args.loss_dev > 0:
                loss *= args.loss_dev
            output_sm = nn.functional.sigmoid(output)
            #output_sm = output
            if torch.isnan(loss):
                saveflag = True
            
            # record loss
            losses.update(loss.item(), proteins[0].size(0))

            # save some data
            # output_sm = nn.functional.sigmoid(output)
            _item = torch.cat((output_sm.detach().cpu(), label.detach().cpu()), 1)
            # del output_sm
            # del target
            saved_data.append(_item)

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                progress.display(i, logger)

        #logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()
        loss_avg, = map(
            _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
            [losses]
        )
        print('saved_data shape = ', len(saved_data), ', ', saved_data[0].shape)
        # import ipdb; ipdb.set_trace()
        # calculate mAP
        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
        np.savetxt(os.path.join(args.output, saved_name), saved_data)
        if dist.get_world_size() > 1:
            dist.barrier()
        
        
        if dist.get_rank() == 0:
            #filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            metric_func = evaluate_performance                
            #mAP, aps = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist], args.num_class, return_each=True)
            y_score = saved_data[:, 0:(saved_data.shape[1] // 2)]
            labels_val = saved_data[:, (saved_data.shape[1] // 2):]
            perf = metric_func(labels_val, y_score, (y_score > threshold).astype(int))
            if confusion and args.comet:
                experiment.log_confusion_matrix(labels_val, (y_score > threshold), step = args.current_epoch, 
                        max_categories=args.num_class, max_examples_per_cell=labels_val.shape[0],
                        row_label=f"Actual GO {args.aspect}", column_label=f"Predicted GO {args.aspect}")

            print('m-aupr, M-aupr, F1, M-F1, acc, Fmax')
            print('%0.5f %0.5f %0.5f %0.5f %0.5f %0.5f\n' % (perf['m-aupr'], perf['M-aupr'], perf['F1'], perf['M-F1'], perf['acc'], perf['Fmax']))

            
            #logger.info(" m_aupr: {}, M_aupr: {}, fmax: {}, acc: {}, ".format(perf['m-aupr'], perf['M-aupr'], perf['F1'], perf['acc']))
        else:
            perf = 0

        if dist.get_world_size() > 1:
            dist.barrier()

    return loss_avg, perf


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


if __name__ == '__main__':
    main()
