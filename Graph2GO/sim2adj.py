import argparse
import numpy as np
import networkx as nx
import scipy.io as sio
import argparse
import os
from scipy import sparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--org', type=str, default='human')
parser.add_argument('--input', type=str, default='sim_result_human.txt')
args = parser.parse_args()

graph = nx.Graph(name='adj')

fRead = open(args.input, 'r')
for i, line in tqdm(enumerate(fRead)):
    splitted = line.strip().split('\t')
    prot1 = splitted[0]
    prot2 = splitted[1]
    evalue = float(splitted[10])
    score = float(splitted[11])
    if not graph.has_node(prot1):
        graph.add_node(prot1)
    if not graph.has_node(prot2):
        graph.add_node(prot2)
    if evalue < 1e-4 :
        graph.add_edge(prot1, prot2, weight=score)
fRead.close()

prot_id = np.genfromtxt(f'../data/{args.org}_protein_id.txt', dtype=str, skip_header=1)

A = nx.adjacency_matrix(graph, nodelist=prot_id)
A = A.todense()
A = np.squeeze(np.asarray(A))
if A.min() < 0:
    print ("### Negative entries in the matrix are not allowed!")
    A[A < 0] = 0
    print ("### Matrix converted to nonnegative matrix.")
if (A.T == A).all():
    pass
else:
    print ("### Matrix not symmetric!")
    A = A + A.T

A = A - np.diag(np.diag(A))
A = A + np.diag(A.sum(axis=1) == 0)
sio.savemat(f'SeqSim_{args.org}.mat', {'Net':sparse.csc_matrix(A)})
