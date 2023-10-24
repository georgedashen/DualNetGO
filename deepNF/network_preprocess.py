# -*- coding: utf-8 -*-
# --------------------------------------------------------
# part of code borrowed from deepNF
# Further revised by Zhuoyang CHEN for different dataset
# --------------------------------------------------------
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import minmax_scale
import argparse
import time
from scipy import sparse

def scaleSimMat(A):
    """Scale rows of similarity matrix"""
    A = A - np.diag(np.diag(A))
    A = A + np.diag(np.array(A.sum(axis=1)).flatten()==0)
    A = A / A.sum(axis=1)

    return A


def RWR(A, K=3, alpha=0.98):
    """Random Walk on graph"""
    A = scaleSimMat(A)
    # Random surfing
    n = A.shape[0]
    P0 = np.eye(n, dtype=float)
    P = P0.copy()
    M = np.zeros((n, n), dtype=float)
    for i in range(0, K):
        P = alpha*np.dot(P, A) + (1. - alpha)*P0
        M = M + P

    return M


def PPMI_matrix(M):
    """ Compute Positive Pointwise Mutual Information Matrix"""
    M = scaleSimMat(M)
    n = M.shape[0]
    col = np.asarray(M.sum(axis=0), dtype=float)
    col = col.reshape((1, n))
    row = np.asarray(M.sum(axis=1), dtype=float)
    row = row.reshape((n, 1))
    D = np.sum(col)

    np.seterr(all='ignore')
    PPMI = np.log(np.divide(D*M, np.dot(row, col)))
    PPMI[np.isnan(PPMI)] = 0
    PPMI[PPMI < 0] = 0

    return PPMI


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = '../data', help = "the data path")
    parser.add_argument('--org', type = str, default = 'human')
    parser.add_argument('--evidence', type = str, default = 'combined', help = "type of PPI to generate")
    args = parser.parse_args()

    assert args.evidence in ['neighborhood', 'fusion', 'cooccurence', 'coexpression', 'experimental', 'database', 'textmining', 'combined'], "Wrong PPI type!"
    
    # Load STRING networks
    Net = sio.loadmat(f'{args.data_path}/{args.org}_net_{args.evidence}.mat', squeeze_me=True)
    Net = Net['Net'].todense()
    t0 = time.time()

    print('Performing RWR...')
    Net = RWR(Net)
    t1 = time.time() - t0
    print(f'RWR took {t1}s')
    
    t0 = time.time()
    print('Caculating PPMI matrix...')
    Net = minmax_scale(PPMI_matrix(Net))
    t2 = time.time() - t0
    print(f'PPMI took {t2}s')

    print ("### Writing output to file...")
    save_file = args.data_path + '/' + args.org + '_net_' + args.evidence + '_RWR_PPMI.mat'
    sio.savemat(save_file, {'Net':sparse.csc_matrix(Net)})
