import argparse
import torch
import numpy as np
from sklearn.preprocessing import minmax_scale
import scipy.io as sio
import networkx as nx

parser = argparse.ArgumentParser('GAE embedding generator for ppi')
parser.add_argument('--org', default='human', help='organism')
parser.add_argument('--evidence', type=str, default='neighborhood')
args = parser.parse_args()

adj = sio.loadmat(f'../data/{args.org}/{args.org}_net_{args.evidence}.mat', squeeze_me=True)
adj = adj['Net'].todense()
adj = minmax_scale(adj)
#edge_index = torch.tensor(adj).nonzero().t().contiguous()

D = nx.DiGraph(adj)
nx.write_weighted_edgelist(D, f'data/network/{args.org}_{args.evidence}_edgeList.txt')
