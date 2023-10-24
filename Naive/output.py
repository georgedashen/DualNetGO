##############################################
## Written by Zhuoyang CHEN
## Date: 2023-08-28
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
example_usage = 'python output.py --aspect P'

parser = argparse.ArgumentParser(description='Naive baseline method', epilog=example_usage)
parser.add_argument('--seed', type=int, default=5959, help='Random seed.')
parser.add_argument('--org', type=str, default='human', help='species')
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

outs = np.mean(labels_train, axis=0)
outs = np.array([outs,]*labels_test.shape[0])
perf = evaluate_performance(labels_test, outs, (outs > 0.5).astype(int))

training_time = time.time() - t_total
print("Train time: {:.4f}s".format(training_time))
print(f"Test accuracy: {perf['Fmax']:.2f}")

fn = f'results_AUPR.csv'
with open(fn, 'a') as f:
    csv.writer(f).writerow([args.org, args.aspect, perf['Fmax'], perf['F1'], perf['M-F1'], perf['acc'], perf['m-aupr'], perf['M-aupr'], perf['M-aupr-labels']])
