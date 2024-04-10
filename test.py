import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import csv
import os

exec(open('cafa_utils.py').read())
exec(open('cafa_evaluate.py').read())
#ext = compile(open('evaluate.py').read(), 'evaluate.py', 'exec')
#exec(ext)

parser = argparse.ArgumentParser(description='evaluate TransFun results based on CAFA3 dataset')
parser.add_argument('--aspect', type=str, default='mf', choices=['mf','bp','cc'], help='function category')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--npy', type=str, default='')
parser.add_argument('--ensemble', action='store_true', default=False)
parser.add_argument('--alpha', type=float, default=0.5)
args = parser.parse_args()

#make matrix
'''
result = pd.read_csv(f'result_cafa3_{args.aspect}_converted.txt', sep=' ', header=None)
if args.test:
    result = result[:100]
test_id = pd.read_csv(f'../{args.aspect}_cafa3_id.txt', header=None)
test_protein = test_id.iloc[:,0].values
'''
label_name, labels_test = generate_label(f'data/{args.aspect}-test.csv')
y_test = labels_test
_, y_val = generate_label(f'data/{args.aspect}-valid.csv')
_, y_train = generate_label(f'data/{args.aspect}-train.csv')
ontologies_names = label_name
#df = np.zeros((len(test_protein),len(label_name)))
#assign values
if args.test:
    if args.npy:
        mat = np.load(args.npy)
    else:
        mat = np.load(f'{args.aspect}-predict.npy')
else:
    df = pd.DataFrame(df, index=test_protein, columns=label_name)
    for i in tqdm(range(len(result))):
        if result.iloc[i][0] in test_protein and result.iloc[i][1] in label_name:
            df.loc[result.iloc[i][0], result.iloc[i][1]] = result.iloc[i][2]
    mat = df.values
    np.save(f'{args.aspect}-predict.npy',mat)

if args.ensemble:
    blast = np.load(f'{args.aspect}-blast.npy')
    mat = mat * args.alpha + blast * (1-args.alpha) * mat.max()

#evaluate
aspect_dict = {'mf':'molecular_function', 'bp':'biological_process', 'cc':'cellular_component'}
ontology = generate_ontology('data/go.obo', specific_space=True, name_specific_space=aspect_dict[args.aspect])
accuracy_data = evaluate(mat, labels_test)

fn = f'result_{args.aspect}_cafa3.csv'
if not os.path.exists(fn):
    with open(fn, 'w') as f:
        csv.writer(f).writerow(['Fmax', 'AuPR', 'IAuPR', 'Smin', 'threshold'])
    with open(fn, 'a') as f:
        csv.writer(f).writerow([accuracy_data['Fmax'], accuracy_data['AuPRC'], accuracy_data['IAuPRC'], accuracy_data['Smin'], accuracy_data['threshold']])
else:
    with open(fn, 'a') as f:
        csv.writer(f).writerow([accuracy_data['Fmax'], accuracy_data['AuPRC'], accuracy_data['IAuPRC'], accuracy_data['Smin'], accuracy_data['threshold']])
