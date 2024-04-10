import pandas as pd
import numpy as np

species_list = ["10090","10116","160488","224308","237561","284812","3702","44689","4932","7227","7955","511145","8355","9606","99287"]

for aspect in ["bp","mf","cc"]:
    for dataset in ["train","valid"]:
        #get entry and taxo
        bp_taxo = pd.read_csv(f"../data/cafa3/{aspect}-{dataset}-taxo.tsv",sep='\t',dtype={'Organism (ID)':str})
        bp_taxo = bp_taxo[~bp_taxo['From'].duplicated()]
        bp_train = pd.read_csv(f"../data/cafa3/{aspect}-{dataset}.csv")
        bp_train = pd.merge(bp_train, bp_taxo[['From','Organism (ID)']],left_on='proteins',right_on='From',how='inner',sort=False)
        bp_train = bp_train[bp_train['Organism (ID)'].isin(species_list)]
        bp_train.sort_values(by='Organism (ID)',inplace=True)
    
        for s in species_list:
            uniprot_t = pd.read_csv(f'../data/cafa3/{s}/uniprotkb_reviewed_true_AND_taxonomy_id_{s}.tsv',sep='\t')
            bp_train.loc[bp_train['Organism (ID)']==s, 'string'] = bp_train.loc[bp_train['Organism (ID)']==s,'proteins'].map(uniprot_t.set_index('Entry')['STRING'])
 
        bp_train['string'] = bp_train['string'].str.replace(';','')
        bp_train.dropna(subset=['string'],inplace=True)
        bp_train['index'] = np.nan

        for s in species_list:
            string_t = pd.read_csv(f"../data/cafa3/{s}/{s}_proteins_id.txt", header = None)
            string_set = bp_train['string'].isin(string_t.iloc[:,0])
            taxo_set = (bp_train['Organism (ID)']==s)
            bp_train.loc[taxo_set.values & string_set.values,'index'] = string_t.reset_index().groupby(0)['index'].first()[bp_train[taxo_set.values & string_set.values]['string'].values].values

        bp_train.dropna(subset=['index'],inplace=True)
        bp_train['index'] = bp_train['index'].astype(int)
        bp_train.to_csv(f"../data/cafa3/{aspect}_go_{dataset}_index.csv", index=False)
