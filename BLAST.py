##############################################
## Written by Zhuoyang CHEN
## Date: 2023-08-18
#############################################

import time
import random
import argparse
import numpy as np
import pandas as pd

example_usage = 'python BLAST.py --aspect mf'

parser = argparse.ArgumentParser(description='BLAST baseline method', epilog=example_usage)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--aspect', type=str, default='mf', help='function category')
parser.add_argument('--fasta', type=str, default='', help='fasta file as input')
parser.add_argument('--txt', type=str, default='', help='if you already perform blast against the training set')
parser.add_argument('--resultdir', type=str, default='')
parser.add_argument('--out', type=str, default='')

args = parser.parse_args()

t_total = time.time()
df_train = pd.read_csv(f'data/cafa3/{args.aspect}-train.csv')
df_test = pd.read_csv(f'data/cafa3/{args.aspect}-test.csv')
prot_id = df_train.iloc[:,0].values
test_id = df_test.iloc[:,0].values
n_class_dict = {'bp':3992, 'mf':677, 'cc':551}
num_labels = n_class_dict[args.aspect]
labels = df_train.iloc[:,2:num_labels+2].to_numpy()
outs = np.zeros((len(test_id),num_labels))


if args.fasta and not args.txt:
    cmd_db = f'makeblastdb -in data/cafa3/{args.aspect}-train.fasta -dbtype prot -out data/cafa3/cafa3_{args.aspect}_train_prot_set'
    if os.path.isfile(f'data/cafa3/cafa3_{args.aspect}_train_prot_set.phr') and os.path.isfile(f'data/cafa3/cafa3_{args.aspect}_train_prot_set.pin') and os.path.isfile(f'data/cafa3/cafa3_{args.aspect}_train_prot_set.psq'):
        pass
    else:
        print('Building protein library for blastp ... ')
        os.system(cmd_db)

    # to ensure 100% output, we do not implement a E-value threshold here, so false positives allowed
    cmd_blast = f'blastp -db data/cafa3/cafa3_{args.aspect}_train_prot_set -query {args.fasta} -out {args.resultdir}/{args.aspect}_homo_query_results.txt -outfmt 6 -max_target_seqs 1'
    print('Querying protein library using blastp ... ')
    os.system(cmd_blast)


print('Parsing query results ...')
query_id = []
if args.txt:
    fRead = open(f'{args.txt}','r')
else:
    fRead = open(f'{args.resultdir}/{args.aspect}_homo_query_results.txt', 'r')
for line in fRead:
    splitted = line.strip().split('\t')
    query = splitted[0]
    if query not in query_id:
        query_id.append(query)
        target = splitted[1]
        target_idx = np.where(prot_id == target)
        query_idx = np.where(test_id == query)
        identity = float(splitted[2]) / 100
        outs[query_idx,:] = labels[target_idx,:] * identity

if args.out:
    np.save(f'{args.resultdir}/{args.out}', outs)
else:
    np.save(f'{args.resultdir}/{args.aspect}-blast.npy', outs)

training_time = time.time() - t_total
print("Train time: {:.4f}s".format(training_time))
