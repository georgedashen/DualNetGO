import argparse
import torch
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAE
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
import scipy.io as sio
import random
import sys
import pickle

parser = argparse.ArgumentParser('GAE embedding generator for ppi')
parser.add_argument('--org', default='human', help='organism')
parser.add_argument('--dataset_dir', default='../data', help='dir of dataset')
parser.add_argument('--evidence', default='combined', choices = ['neighborhood', 'fusion', 'cooccurence', 'coexpression', 'experimental', 'database', 'textmining', 'combined'], help='what evidence is used to construct the PPI graph')
parser.add_argument('--epoch', type=int, default=20)
args = parser.parse_args()

adj = sio.loadmat(f'{args.dataset_dir}/{args.org}/{args.org}_net_{args.evidence}.mat', squeeze_me=True)
adj = adj['Net'].todense()
edge_index = torch.tensor(adj).nonzero().t().contiguous()

feature_file = f'{args.dataset_dir}/{args.org}/features.npy'
with open(feature_file, 'rb') as f:
    x = pickle.load(f)
x = torch.tensor(x).float()

def set_rand_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_rand_seed(42)

print(f'PPI density: {len(edge_index[0])/adj.shape[0]**2}')
print(f'feature shape: {x.shape}')

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

model = GAE(GCNEncoder(x.shape[1], 512))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
x = x.to(device)
edge_index = edge_index.to(device)

train_loader = NeighborSampler(edge_index, sizes=[30,15], batch_size=128, shuffle=True, num_workers=32)
final_loader = NeighborSampler(edge_index, sizes=[-1], batch_size=128, shuffle=False, num_workers=32, drop_last=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    total_loss = 0
    for i, (batch_size,n_id,adjs) in enumerate(train_loader):
        optimizer.zero_grad()
        if isinstance(adjs[0],torch.Tensor):
            adjs = adjs[0].to(device) #1-layer
        else:
            adjs = adjs[0][0].to(device)
        z = model.encode(x[n_id,:], adjs)
        loss = model.recon_loss(z, adjs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

for epoch in range(1,args.epoch+1):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

model.eval()
output = torch.zeros((adj.shape[0],512)).to(device)
with torch.no_grad():
    for i, (batch_size,n_id,adjs) in enumerate(final_loader):
        if isinstance(adjs[0],torch.Tensor):
            adjs = adjs[0].to(device)
        else:
            adjs = adjs[0][0].to(device)
        z = model.encode(x[n_id,:], adjs)
        output[n_id[:batch_size]] = z[:batch_size,:]

output = output.detach().cpu().numpy() #should be 19385*512
np.save(f'{args.dataset_dir}/{args.org}/{args.org}_net_{args.evidence}_GAE.npy', output)
