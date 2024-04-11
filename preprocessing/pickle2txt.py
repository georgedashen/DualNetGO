import sys
import pickle

#s = sys.argv[1]

species_list = ['10090','10116','160488','224308','237561','284812','3702','44689','4932','7227','7955','511145','8355','9606','99287']

for s in species_list:
    string_id = pickle.load(open(f'../data/cafa3/{s}/{s}_proteins.txt','rb'))
    with open(f'../data/cafa3/{s}/{s}_proteins_id.txt','w') as f:
        for item in string_id:
            f.write(f'{item}\n')
