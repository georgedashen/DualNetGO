import argparse
from Bio import SeqIO
import scipy.io as sio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--org', default='human', type=str, help='species')
parser.add_argument('--aspect', default='P', type=str)
args = parser.parse_args()

org_code_dict = {'human':'9606', 'mouse':'10090'}
org_code = org_code_dict[args.org]

Annot = sio.loadmat(f'../data/{args.org}/{args.org}_annot.mat', squeeze_me=True)
train_idx = Annot['indx'][args.aspect].tolist()['train'].tolist().tolist()
valid_idx = Annot['indx'][args.aspect].tolist()['valid'].tolist().tolist()
test_idx = Annot['indx'][args.aspect].tolist()['test'].tolist().tolist()

prot_id = np.genfromtxt(f'../data/{args.org}_protein_id.txt', dtype=str, skip_header=1)

train = set(prot_id[train_idx])
test = set(prot_id[test_idx])

with open(f'./{args.org}_{args.aspect}_train.fa', 'w') as fout:
    for record in SeqIO.parse(f'../data/{org_code}.protein.sequences.v11.5.fa','fasta'):
        if record.id in train:
            SeqIO.write(record, fout, 'fasta')
with open(f'./{args.org}_{args.aspect}_test.fa', 'w') as fout:
    for record in SeqIO.parse(f'../data/{org_code}.protein.sequences.v11.5.fa','fasta'):
        if record.id in test:
            SeqIO.write(record, fout, 'fasta')
