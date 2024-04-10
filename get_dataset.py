import torch
from scipy import sparse
from torch.utils.data import Dataset

class multimodesDataset(Dataset):
    def __init__(self, num_modes, modes_features, labels):
        self.modes_features = modes_features
        self.labels = labels
        self.num_modes = num_modes

    def __getitem__(self, index):
        modes_features = []
        for i in range(self.num_modes):
            modes_features.append(self.modes_features[i][index,:])
        return modes_features, self.labels[index,:]

    def __len__(self):
        return self.modes_features[7].size(0)

    def subset(self, index):
        return torch.utils.data.Subset(self, index)
