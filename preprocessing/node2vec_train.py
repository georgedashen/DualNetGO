import argparse
import torch
import numpy as np
from torch_geometric.nn import Node2Vec
import scipy.io as sio
from tqdm import tqdm

parser = argparse.ArgumentParser('node2vec embedding generator for ppi')
parser.add_argument('--org', default='human', help='organism')
parser.add_argument('--dataset_dir', default='../data', help='dir of dataset')
parser.add_argument('--evidence', default='combined', choices = ['neighborhood','fusion','cooccurence', 'coexpression', 'experimental', 'database', 'textmining', 'combined'], help='what evidence is used to construct the PPI graph')
parser.add_argument('--epoch', type=int, default=20)
args = parser.parse_args()

adj = sio.loadmat(f'{args.dataset_dir}/{args.org}/{args.org}_net_'+args.evidence+'.mat', squeeze_me=True)
adj = adj['Net'].todense()
edge_index = torch.tensor(adj).nonzero().t().contiguous()

def set_rand_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_rand_seed(42)

print(f'PPI density: {len(edge_index[0])/adj.shape[0]**2}')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = Node2Vec(edge_index, embedding_dim=512, walk_length=20,
             context_size=10, walks_per_node=10, num_nodes=adj.shape[0],
             num_negative_samples=1, p=1, q=1, sparse=True).to(device)

loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

def train():
    model.train()  # put model in train model
    total_loss = 0
    for pos_rw, neg_rw in tqdm(loader):
        optimizer.zero_grad()  # set the gradients to 0
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))  # compute the loss for the batch
        loss.backward()
        optimizer.step()  # optimize the parameters
        total_loss += loss.item()
    return total_loss / len(loader)


for epoch in range(1, 20):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

model.eval()
output = model().cpu().detach().numpy() #should be 19385*512
np.save(f'{args.dataset_dir}/{args.org}/{args.org}_net_'+args.evidence+'_node2vec.npy', output)
