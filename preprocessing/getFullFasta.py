import numpy as np
import pandas as pd
from Bio import SeqIO
import gzip

species_list = ['10090','10116','160488','224308','237561','284812','3702','44689','4932','7227','7955','511145','8355','9606','99287']

# for each species, string id that only in ppi protein_id
# a dataframe with all protein id, taxo
# blast result target get one taxo, use this to filter df, map target as index
# return taxo and index in order
string_ids = []
taxo = []
for s in species_list:
    ids = np.genfromtxt(f'../data/cafa3/{s}/{s}_proteins_id.txt', dtype=str)
    string_ids = string_ids + ids.tolist()
    taxo = taxo + ([s]*len(ids))
    with open('../Dataset/all_cafa_string.fa', 'a') as fout:
        for record in SeqIO.parse(gzip.open(f'../data/cafa3/{s}/{s}.protein.sequences.v12.0.fa.gz','rt'),'fasta'):
            if record.id in ids:
                SeqIO.write(record, fout, 'fasta')

df = pd.DataFrame({'string':string_ids,'taxo':taxo})
df.to_csv('../dataset/cafa3/all_proteins_id.csv',index=False)

