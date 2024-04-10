#############################
## Revised by Zhuoyang CHEN
## Date: 2024/03/24
## Content: generate esm-2 embedding for string fasta
############################

import argparse
import os
import pandas as pd
import pickle
from Bio import SeqIO
import gzip
import numpy as np
from tqdm import tqdm
import ktrain
from ktrain import text
from transformers import *
import tensorflow as tf
import time
from npy_append_array import NpyAppendArray

ext = compile(open('utils_cafa.py').read(), 'utils.cafa.py', 'exec')
exec(ext)
#from utils import *

example_usage = 'python extract-embeddings.py --org 9606 --model_name Esm2'
parser = argparse.ArgumentParser(description='extract embeddings from pLM', epilog=example_usage)
parser.add_argument('--org', type=str, default='9606')
parser.add_argument('--fasta', default=None)
parser.add_argument('--maxlen', type=int, default=500, help='max sequence len before truncating')
parser.add_argument('--overlap', type=int, default=250, help='overlap sequence')
parser.add_argument('--model_dir', type=str)
parser.add_argument('--model_name', type=str, default='Esm2', help='base model')
parser.add_argument('--batch', type=int, default=128, help='batch to encode')
parser.add_argument('--test', action='store_true', default=False, help='enter test mode to load a small subset of data')
args = parser.parse_args()


args.model_name = f'{args.model_dir}/{args.model_name}'

# Load ppi protein name in order
with open(f'../data/cafa3/{args.org}/{args.org}_proteins.txt', 'rb') as f:
    prot_id = pickle.load(f)

# Load fasta data
ids = []
seqs = []
if args.fasta:
    fasta = str(args.fasta)
else:
    fasta = f'../data/cafa3/{args.org}/{args.org}.protein.sequences.v12.0.fa.gz'

for record in SeqIO.parse(gzip.open(fasta,'rt'),'fasta'):
    if record.id in prot_id:
        ids.append(record.id)
        seqs.append(str(record.seq))

df_train = pd.DataFrame({'protein':ids, 'sequences':seqs})
df_train = df_train.set_index('protein').loc[prot_id].reset_index()    

# Generate slices from sliding window technique
X_train, positions_train = generate_data(df_train, subseq=args.maxlen, overlap=args.overlap)
y_train = np.zeros(len(X_train))

################ Training #####################
model, tokenizer, ds = get_model(args.model_name)
file_path = f'../data/cafa3/{args.org}/{args.org}_Esm2.npy'
t0 = time.time()
get_embeddings(ds, model, tokenizer, file_path)
embed = np.load(file_path)
X_train = protein_embedding(embed, positions_train)
np.save(file_path, X_train)
print(f'Embedding time for train: {time.time()-t0}')


