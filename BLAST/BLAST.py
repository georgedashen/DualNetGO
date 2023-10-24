##############################################
## Written by Zhuoyang CHEN
## Date: 2023-08-18
#############################################

from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import scipy.io as sio
import numpy as np
import warnings
import csv
from validation import evaluate_performance

warnings.filterwarnings("ignore")
example_usage = 'python BLAST.py --aspect P'

parser = argparse.ArgumentParser(description='BLAST baseline method', epilog=example_usage)
parser.add_argument('--seed', type=int, default=5959, help='Random seed.')
parser.add_argument('--org', type=str, default='human')
parser.add_argument('--aspect', type=str, default='P', help='function category')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

t_total = time.time()

Annot = sio.loadmat(f'../data/{args.org}_annot.mat', squeeze_me=True)
train_idx = Annot['indx'][args.aspect].tolist()['train'].tolist().tolist()
valid_idx = Annot['indx'][args.aspect].tolist()['valid'].tolist().tolist()
test_idx = Annot['indx'][args.aspect].tolist()['test'].tolist().tolist()

labels_train = np.array(Annot['GO'][args.aspect].tolist()['train'].tolist())
labels_valid = np.array(Annot['GO'][args.aspect].tolist()['valid'].tolist())
labels_test = np.array(Annot['GO'][args.aspect].tolist()['test'].tolist())

prot_id = np.genfromtxt(f'../data/{args.org}_protein_id.txt', dtype=str, skip_header=1)

labels = np.zeros((prot_id.shape[0],labels_test.shape[1]))
labels[train_idx,:] = labels_train
labels[valid_idx,:] = labels_valid
labels[test_idx,:] = labels_test
outs = np.zeros(labels.shape)

fRead = open(f'blast_result_{args.org}_{args.aspect}.txt', 'r')
#fRead.readline()
for line in fRead:
    splitted = line.strip().split('\t')
    query = splitted[0]
    query_idx = np.where(prot_id == query)
    target = splitted[1]
    target_idx = np.where(prot_id == target)
    identity = float(splitted[2]) / 100
    outs[query_idx,:] = labels[target_idx,:] * identity

outs = outs[test_idx,:]
perf = evaluate_performance(labels_test, outs, (outs > 0.5).astype(int))


training_time = time.time() - t_total
print("Train time: {:.4f}s".format(training_time))
print(f"Test accuracy: {perf['Fmax']:.2f}")

fn = f'{args.org}_results.csv'
with open(fn, 'a') as f:
    csv.writer(f).writerow([args.aspect, perf['Fmax'], perf['F1'], perf['M-F1'], perf['acc'], perf['m-aupr'], perf['M-aupr']])
