import sys
import pickle

s = sys.argv[1]

string_id = pickle.load(open(f'../data/cafa3/{s}/{s}_proteins.txt','rb'))
with open(f'../data/cafa3/{s}/{s}_proteins_id.txt','w') as f:
    for item in string_id:
        f.write(f'{item}\n')
