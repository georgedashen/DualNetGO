#revised by Zhuoyang CHEN to return train, valid, test instead of combining train/valid together

import os
import scipy.io as sio
import torch
from torch.utils.data import Dataset


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
    #=========load PPMIs========
    Y = []
    for e in ['neighborhood', 'fusion', 'cooccurence', 'coexpression', 'experimental', 'database', 'textmining']:
        fn = f'{args.org}_net_{e}_RWR_PPMI.mat'
        y = sio.loadmat(os.path.join(args.dataset_dir, fn), squeeze_me=True)
        y = y['Net'].todense()
        Y.append(torch.from_numpy(y).float())
    
    modefeature_lens = Y[0].shape[1]

    full_dataset = multimodesFullDataset(7, Y)

    return full_dataset, modefeature_lens



