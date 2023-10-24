import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.utils import resample

import argparse
import time
import scipy.io as sio
from validation import evaluate_performance
import csv

def kernel_func(X, Y=None, param=0):
    if param != 0:
        K = rbf_kernel(X, Y, gamma=param)
    else:
        K = linear_kernel(X, Y)

    return K

def temporal_holdout(X_train,y_train, X_valid,y_valid, X_test,y_test, ker='rbf'):
    # range of hyperparameters
    C_range = 10.**np.arange(-1, 3)
    if ker == 'rbf':
        gamma_range = 10.**np.arange(-3, 1)
    elif ker == 'lin':
        gamma_range = [0]
    else:
        print ("### Wrong kernel.")

    # pre-generating kernels
    print ("### Pregenerating kernels...")
    K_rbf_train = {}
    K_rbf_test = {}
    K_rbf_valid = {}
    for gamma in gamma_range:
        K_rbf_train[gamma] = kernel_func(X_train, param=gamma)
        K_rbf_test[gamma] = kernel_func(X_test, X_train, param=gamma)
        K_rbf_valid[gamma] = kernel_func(X_valid, X_train, param=gamma)
    print ("### Done.")
    print ("Train samples=%d; #Test samples=%d" % (y_train.shape[0], y_test.shape[0]))

    # parameter fitting
    C_opt = None
    gamma_opt = None
    max_Fmax = 0
    tmax = 0
    for C in C_range:
        for gamma in gamma_range:
            # Multi-label classification
            clf = OneVsRestClassifier(svm.SVC(C=C, kernel='precomputed',
                                              random_state=123,
                                              probability=True), n_jobs=-1)
            clf.fit(K_rbf_train[gamma], y_train)
            # y_score_valid = clf.decision_function(K_rbf_valid[gamma])
            y_score_valid = clf.predict_proba(K_rbf_valid[gamma])
            #y_pred_valid = clf.predict(K_rbf_valid[gamma])
            perf = evaluate_performance(y_valid,
                                        y_score_valid,
                                        (y_score_valid>0.5).astype(int))
            cur_Fmax = perf['Fmax']
            print ("### gamma = %0.3f, C = %0.3f, Fmax = %0.3f" % (gamma, C, cur_Fmax))
            if cur_Fmax > max_Fmax:
                C_opt = C
                gamma_opt = gamma
                max_Fmax = cur_Fmax
                tmax = perf['tmax']
    print ("### Optimal parameters: ")
    print ("C_opt = %0.3f, gamma_opt = %0.3f" % (C_opt, gamma_opt))
    print ("### Train dataset: Fmax = %0.3f" % (max_Fmax))
    print
    print ("### Computing performance on test dataset...")
    clf = OneVsRestClassifier(svm.SVC(C=C_opt, kernel='precomputed',
                                      random_state=123,
                                      probability=True), n_jobs=-1)
    clf.fit(K_rbf_train[gamma_opt], y_train)

    # Compute performance on test set
    # y_score = clf.decision_function(K_rbf_test[gamma_opt])
    y_score = clf.predict_proba(K_rbf_test[gamma_opt])
    #y_pred = clf.predict(K_rbf_test[gamma_opt])

    perf = evaluate_performance(y_test, y_score, (y_score>tmax).astype(int))

    return perf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--org', type=str, default='human')
    parser.add_argument('--aspect', type=str, default='P')
    parser.add_argument('--out', type=str, default='results.csv')
    args = parser.parse_args()
    
    X = sio.loadmat(f'{args.org}_Mashup.mat') #dim=800
    X = X['x'].T
    Annot = sio.loadmat(f'../data/{args.org}_annot.mat', squeeze_me=True)

    train_idx = Annot['indx'][args.aspect].tolist()['train'].tolist().tolist()
    valid_idx = Annot['indx'][args.aspect].tolist()['valid'].tolist().tolist()
    test_idx = Annot['indx'][args.aspect].tolist()['test'].tolist().tolist()

    X_train = X[train_idx]
    y_train = np.array(Annot['GO'][args.aspect].tolist()['train'].tolist())
    X_valid = X[valid_idx]
    y_valid = np.array(Annot['GO'][args.aspect].tolist()['valid'].tolist())
    X_test = X[test_idx]
    y_test = np.array(Annot['GO'][args.aspect].tolist()['test'].tolist())
    
    t0 = time.time()
    perf = temporal_holdout(X_train,y_train,X_valid,y_valid,X_test,y_test)
    print(f'Training time: {time.time()-t0}')

    fn = f'{args.out}'
    with open(fn, 'a') as f:
        csv.writer(f).writerow([args.org, args.aspect, perf['Fmax'], perf['F1'], perf['M-F1'], perf['acc'], perf['m-aupr'], perf['M-aupr']])

    print(perf)





