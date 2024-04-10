import sys
import scipy.io as sio
import numpy as np
import pandas as pd

# python getFullSet.py mf train

aspect = sys.argv[1] # bp, mf, cc
dataset = sys.argv[2] # train, valid

df = pd.read_csv(f'../data/cafa3/{aspect}_go_{dataset}_index.csv')
df = df.dropna()
species_list = df['taxo'].unique()

for e in ['neighborhood', 'fusion', 'cooccurence', 'coexpression', 'experimental', 'database', 'textmining']:
    net = np.empty((0,512))
    for s in species_list:
        df_t = df[df['taxo']==s]
        fn = f'../data/cafa3/{s}/{s}_net_{e}_AE.npy'
        y = np.load(fn)
        net = np.vstack((net,y[np.array(df_t['index'].values,dtype=int),:]))
    np.save(f'../data/cafa3/{aspect}_{dataset}_all_net_{e}_AE.npy', net)

esm2 = np.empty((0,1280))
for s in species_list:
    df_t = df[df['taxo']==s]
    fe = f'../data/cafa3/{s}/{s}_Esm2.npy'
    y = np.load(fe)
    esm2 = np.vstack((esm2,y[np.array(df_t['index'].values,dtype=int),:]))
np.save(f'../data/cafa3/{aspect}_{dataset}_all_Esm2.npy', esm2)

label_dict = {'bp':3992, 'mf':677, 'cc':551}
'''
y = []
for i in range(len(df)):
    y.append(df.iloc[i, 2:label_dict[aspect]+2])

np.save(f'../Dataset/{aspect}-{dataset}-labels.npy', np.array(y))
'''

