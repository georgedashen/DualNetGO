# If error occurs related to proxy, you can use commands:
# unset http_proxy
# unset https_proxy

import argparse
import torch
import numpy as np
import pickle
from sklearn.preprocessing import minmax_scale
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAE
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import scipy.io as sio
import random
import time
import sys

parser = argparse.ArgumentParser('GAE embedding generator for ppi')
parser.add_argument('--org', default='human', help='organism')
parser.add_argument('--input', type=str, default='SeqSim_human.mat')
parser.add_argument('--type', type=str, default='sim', choices=['adj','sim'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save', action='store_true', default=False)
args = parser.parse_args()

adj = sio.loadmat(args.input, squeeze_me=True)
adj = adj['Net'].todense()
edge_index = torch.tensor(adj).nonzero().t().contiguous()

if args.type == 'adj':
    with open(f'../data/{args.org}/features.npy', 'rb') as f:
        features = pickle.load(f)
    CT = np.load(f'{args.org}_CT_343.npy')
    CT = minmax_scale(CT)
    x = torch.cat([torch.tensor(features).float(), torch.tensor(CT).float()],1)
elif args.type == 'sim':
    with open(f'../data/{args.org}/features.npy', 'rb') as f:
        x = pickle.load(f)

x = torch.tensor(x).float()

def set_rand_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_rand_seed(args.seed)

print(f'PPI density: {len(edge_index[0])/adj.shape[0]**2}')
print(f'feature shape: {x.shape}')

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 400, cached=False) # cached only for transductive learning
        self.conv2 = GCNConv(400, 800, cached=False) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

model = GAE(GCNEncoder(x.shape[1]))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

data = Data(x=x, edge_index=edge_index)

train_loader = NeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30,15],
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            shuffle=True
        )
test_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=128, shuffle=False)

#sampled_data = next(iter(loader))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch.x = batch.x.to(device)
        batch.edge_index = batch.edge_index.to(device)
        z = model.encode(batch.x, batch.edge_index)
        loss = model.recon_loss(z, batch.edge_index)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

t0 = time.time()
for epoch in range(args.epochs):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
print(f'Training time: {time.time()-t0}')

model.eval()
output = torch.zeros((adj.shape[0],800)).to(device)
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        batch.x = batch.x.to(device)
        batch.edge_index = batch.edge_index.to(device)
        z = model.encode(batch.x, batch.edge_index)
        output[batch.input_id] = z[:batch.batch_size,:]

output = output.detach().cpu().numpy() #should be 19385*512
if args.save:
    np.save(f'{args.org}_net_{args.type}_VGAE_epoch{args.epochs}.npy', output)
