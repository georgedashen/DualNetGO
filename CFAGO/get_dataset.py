from sklearn.preprocessing import minmax_scale
from scipy import sparse
import os
import pickle
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset


class multimodesDataset(Dataset):
    def __init__(self, num_modes, modes_features, labels):
        self.modes_features = modes_features
        self.labels = labels
        self.num_modes = num_modes
        
    def __getitem__(self, index):
        modes_features = []
        for i in range(self.num_modes):
            modes_features.append(self.modes_features[i][index])
        return modes_features, self.labels[index]
    
    def __len__(self):
        return self.modes_features[0].size(0)
    
class multimodesFullDataset(Dataset):
    def __init__(self, num_modes, modes_features):
        self.modes_features = modes_features
        self.num_modes = num_modes
        
    def __getitem__(self, index):
        modes_features = []
        for i in range(self.num_modes):
            modes_features.append(self.modes_features[i][index])
        return modes_features
    
    def __len__(self):
        return self.modes_features[0].size(0)

def get_ssl_datasets(args):
    'retrun self-supervised data'
    #=========load feature==========
    if args.org == 'mouse':
        feature_file = args.dataset_dir + '/features_mouse.npy'
    else:
        feature_file = args.dataset_dir + '/features.npy'
    with open(feature_file, 'rb') as f:
        Z = pickle.load(f)
    Z = minmax_scale(np.asarray(Z))
    
    #=========load PPMIs========
    ppmi = args.org + '_net_' + args.evidence + '.mat'
    pf = os.path.join(args.dataset_dir, ppmi)
    N = sio.loadmat(pf, squeeze_me=True)
    X = N['Net'].todense()
    X = minmax_scale(np.asarray(X))
    #X = np.hstack((X,Z))

    full_X = torch.from_numpy(X)
    full_X = full_X.float()
    
    full_Z = torch.from_numpy(Z)
    full_Z = full_Z.float()
    modefeature_lens = [X.shape[1], Z.shape[1]]
    
    full_dataset = multimodesFullDataset(2, [full_X, full_Z])
    
    return full_dataset, modefeature_lens


def get_datasets(args):
    #===========load annot============
    Annot = sio.loadmat(args.dataset_dir + '/' + args.org + '_annot.mat', squeeze_me=True)
    
    #=========load feature==========
    if args.org == 'mouse':
        feature_file = args.dataset_dir + '/features_mouse.npy'
    else:
        feature_file = args.dataset_dir + '/features.npy'
    with open(feature_file, 'rb') as f:
        Z = pickle.load(f)
    Z = minmax_scale(np.asarray(Z))
    
    #=========load PPMIs========
    ppmi = args.org + '_net_' + args.evidence + '.mat'
    pf = os.path.join(args.dataset_dir, ppmi)
    N = sio.loadmat(pf, squeeze_me=True)
    X = N['Net'].todense()
    X = minmax_scale(np.asarray(X))
    #X = np.hstack((X,Z))
    
    #========load label===========
    train_idx = Annot['indx'][args.aspect].tolist()['train'].tolist().tolist()
    valid_idx = Annot['indx'][args.aspect].tolist()['valid'].tolist().tolist()
    test_idx = Annot['indx'][args.aspect].tolist()['test'].tolist().tolist()
    
    X_train = X[train_idx]
    labels_train = np.array(Annot['GO'][args.aspect].tolist()['train'].tolist())
    print('labels_train shape = ', labels_train.shape)
    
    X_valid = X[valid_idx]
    labels_valid = np.array(Annot['GO'][args.aspect].tolist()['valid'].tolist())
    
    X_test = X[test_idx]
    labels_test = np.array(Annot['GO'][args.aspect].tolist()['test'].tolist())
    
    X_train = torch.from_numpy(X_train).float()
    labels_train = torch.from_numpy(labels_train).float()
    
    X_valid = torch.from_numpy(X_valid).float()
    labels_valid = torch.from_numpy(labels_valid).float()
    
    X_test = torch.from_numpy(X_test).float()
    labels_test = torch.from_numpy(labels_test).float()
    
    Z_train = Z[train_idx]
    Z_train = torch.from_numpy(Z_train).float()
    
    Z_valid = Z[valid_idx]
    Z_valid = torch.from_numpy(Z_valid).float()
    
    Z_test = Z[test_idx]
    Z_test = torch.from_numpy(Z_test).float()
    
    train_dataset = multimodesDataset(2, [X_train, Z_train], labels_train)
    valid_dataset = multimodesDataset(2, [X_valid, Z_valid], labels_valid)
    test_dataset  = multimodesDataset(2, [X_test, Z_test], labels_test)
    modefeature_lens = [X_train.shape[1], Z_train.shape[1]]
    print('X_train shape = ', X_train.shape)
    
    return train_dataset, valid_dataset, test_dataset, modefeature_lens
