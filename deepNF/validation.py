# --------------------------------------------------------
# Codes are borrowed from deepNF and Graph2GO
# Further revised by Zhuoyang CHEN for proper evaluation on test set
# There are also some wrong in calculating metrics, like multi-class Fmax and F1, should in macro way not micro
# --------------------------------------------------------

import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression

import numpy as np


def kernel_func(X, Y=None, param=0):
    if param != 0:
        K = rbf_kernel(X, Y, gamma=param)
    else:
        K = linear_kernel(X, Y)

    return K

def calculate_fmax(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max, t_max

def real_AUPR(label, score):
    """Computing real AUPR . By Vlad and Meet"""
    label = label.flatten()
    score = score.flatten()

    order = np.argsort(score)[::-1]
    label = label[order]

    P = np.count_nonzero(label)
    # N = len(label) - P

    TP = np.cumsum(label, dtype=float)
    PP = np.arange(1, len(label)+1, dtype=float)  # python

    x = np.divide(TP, P)  # recall
    y = np.divide(TP, PP)  # precision
    
    pr = np.trapz(y, x)
    f = np.divide(2*x*y, (x + y))
    idx = np.where((x + y) != 0)[0]
    if len(idx) != 0:
        f = np.max(f[idx])
    else:
        f = 0.0 

    return pr, f


def ml_split(y):
    """Split annotations"""
    kf = KFold(n_splits=5, shuffle=True)
    splits = []
    for t_idx, v_idx in kf.split(y):
        splits.append((t_idx, v_idx))

    return splits


def evaluate_performance(y_test, y_score, y_pred):
    """Evaluate performance"""
    n_classes = y_test.shape[1]
    perf = dict()
    # Compute macro-averaged AUPR
    perf["M-aupr"] = 0.0
    perf["M-aupr-labels"] = []
    n = 0
    for i in range(n_classes):
        re, _ = real_AUPR(y_test[:, i], y_score[:, i])
        perf["M-aupr-labels"].append(re)
        if sum(y_test[:, i]) > 0:
            n += 1
            perf["M-aupr"] += re
    perf["M-aupr"] /= n
    #print('number of test labels is: ', n)
    # Compute micro-averaged AUPR
    perf["m-aupr"], _ = real_AUPR(y_test, y_score)

    # Computes accuracy
    #perf['acc'] = accuracy_score(y_test, y_pred)
    acc_count = 0
    for i in range(y_test.shape[0]):
        if (y_test[i,:] == y_pred[i,:]).all():
            acc_count = acc_count + 1
    perf['acc'] = acc_count / y_test.shape[0]

    # Computes F1-score
    alpha = 3
    y_new_pred = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        top_alpha = np.argsort(y_score[i, :])[-alpha:]
        y_new_pred[i, top_alpha] = np.array(alpha*[1])
    perf["F1"] = f1_score(y_test, y_new_pred, average='micro')
    perf["M-F1"] = f1_score(y_test, y_new_pred, average='macro') 
    perf["Fmax"], perf["tmax"] = calculate_fmax(y_score, y_test)
    return perf

