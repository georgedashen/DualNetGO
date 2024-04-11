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
parser.add_argument('--npy', type=str, default='', help='use npy matrix object as input')
parser.add_argument('--txt', type=str, default='', help='to generate matrix from tabular score file')
parser.add_argument('--ensemble', action='store_true', default=False)
parser.add_argument('--blast', type=str, default='', help='blast score saved in npy object')
parser.add_argument('--alpha', default=None)
parser.add_argument('--resultdir', type=str, default='.')
args = parser.parse_args()


alpha_dict = {'bp':0.72, 'mf':0.62, 'cc':0.76}
if args.alpha:
    alpha_dict[args.aspect] = float(args.alpha)
if not args.ensemble:
    alpha_dict[args.aspect] = 1.0
alpha = alpha_dict[args.aspect]


label_name, labels_test = generate_label(f'data/cafa3/{args.aspect}-test.csv')
_, y_val = generate_label(f'data/cafa3/{args.aspect}-valid.csv')
_, y_train = generate_label(f'data/cafa3/{args.aspect}-train.csv')
y_test = labels_test
ontologies_names = label_name

if args.npy:
    mat = np.load(args.npy)
elif args.txt:
    result = pd.read_csv(args.txt, sep='\t', header=None)
    test_protein = pd.read_csv(f'data/cafa3/{args.aspect}-test.csv').iloc[:,0].values
    df = pd.DataFrame(df, index=test_protein, columns=label_name)
    for i in tqdm(range(len(result))):
        if result.iloc[i][0] in test_protein and result.iloc[i][1] in label_name:
            df.loc[result.iloc[i][0], result.iloc[i][1]] = result.iloc[i][2]
    mat = df.values
    np.save(f'{args.resultdir}/{args.aspect}-predict.npy',mat)

if args.ensemble:
    blast = np.load(args.blast)
    mat = mat * alpha + blast * (1-alpha) * mat.max()

#evaluate
aspect_dict = {'mf':'molecular_function', 'bp':'biological_process', 'cc':'cellular_component'}
ontology = generate_ontology('data/cafa3/go.obo', specific_space=True, name_specific_space=aspect_dict[args.aspect])
accuracy_data = evaluate(mat, labels_test)

fn = f'{args.resultdir}/result_{args.aspect}_cafa3_ensemble.csv'
if not os.path.exists(fn):
    with open(fn, 'w') as f:
        csv.writer(f).writerow(['Fmax', 'AuPR', 'IAuPR', 'Smin', 'threshold', 'alpha'])
    with open(fn, 'a') as f:
        csv.writer(f).writerow([accuracy_data['Fmax'], accuracy_data['AuPRC'], accuracy_data['IAuPRC'], accuracy_data['Smin'], accuracy_data['threshold'], alpha])
else:
    with open(fn, 'a') as f:
        csv.writer(f).writerow([accuracy_data['Fmax'], accuracy_data['AuPRC'], accuracy_data['IAuPRC'], accuracy_data['Smin'], accuracy_data['threshold'], alpha])
