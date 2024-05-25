import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from collections import Counter
import math

def evaluate(predictions, ground_truth):
  y_ = np.concatenate([y_train, y_val, y_test])
  annots = []
  for i in y_:
    actual = []
    for j in range(len(i.tolist())):
      if i[j] == 1:
        actual.append(ontologies_names[j])
    annots.append(actual)

  cnt = Counter()
  for x in annots:
    cnt.update(x)

  ic = {}

  for go_id, n in cnt.items():
    if go_id in ontology:
      parents = ontology[go_id]['parents']
      if len(parents) == 0:
        min_n = n
      else:
        min_n = min([cnt[x] for x in parents])
      ic[go_id] = math.log(min_n / n, 2)

  precisions = []
  recalls = []
  f1s = []
  thresholds = []
  f1_max_value = -1
  f1_max_threshold = -1
  smin = 1e100

  for i in tqdm(range(1, 101)):
    threshold = i/100
    p, r = 0, 0
    ru, mi = 0, 0
    number_of_proteins = 0

    for idx_protein in range(len(predictions)):
      protein_pred = set()
      protein_gt = set()

      for idx_term in range(len(ontologies_names)):
        if ground_truth[idx_protein][idx_term] == 1:
          protein_gt.add(ontologies_names[idx_term])
          for parent in ontology[ontologies_names[idx_term]]['ancestors']:
            protein_gt.add(parent)

        if predictions[idx_protein][idx_term] >= threshold:
          protein_pred.add(ontologies_names[idx_term])
          for parent in ontology[ontologies_names[idx_term]]['ancestors']:
            protein_pred.add(parent)
      
      if len(protein_pred) > 0:
        number_of_proteins += 1
        p += len(protein_pred.intersection(protein_gt)) / len(protein_pred)
      r += len(protein_pred.intersection(protein_gt)) / len(protein_gt)

      tp = protein_pred.intersection(protein_gt)
      fp = protein_pred - tp
      fn = protein_gt - tp
      for go_id in fp:
        mi += ic[go_id]
      for go_id in fn:
        ru += ic[go_id]
      

    if number_of_proteins > 0:
      threshold_p = p / number_of_proteins
    else:
      threshold_p = 0

    threshold_r = r / len(predictions)

    precisions.append(threshold_p)
    recalls.append(threshold_r)
    
    f1 = 0
    if threshold_p > 0 or threshold_r > 0:
      f1 = (2 * threshold_p * threshold_r) / (threshold_p + threshold_r)
    
    f1s.append(f1)
    thresholds.append(threshold)

    if f1 > f1_max_value:
      f1_max_value = f1
      f1_max_threshold = threshold

    ru = ru / len(predictions)
    mi = mi / len(predictions)

    smin_atual = math.sqrt((ru * ru) + (mi * mi))

    if smin_atual < smin:
      smin = smin_atual

  print('\nFmax:', f1_max_value)
  print('F1 threshold:', f1_max_threshold)
  print('Smin:', smin)

  precisions = np.array(precisions)
  recalls = np.array(recalls)
  sorted_index = np.argsort(recalls)
  recalls = recalls[sorted_index]
  precisions = precisions[sorted_index]
  aupr = np.trapz(precisions, recalls)
  print('AuPRC:', aupr)

  new_p = []
  new_r = []
  for i in range(1, 101):
    if len(np.where(recalls >= i/100)[0]) != 0:
      idx = np.where(recalls >= i/100)[0][0]
      new_r.append(i/100)
      new_p.append(max(precisions[idx:]))

  iaupr = np.trapz(new_p, new_r)
  print('IAuPRC:', iaupr)

  return {'Fmax':f1_max_value, 'AuPRC':aupr, 'IAuPRC':iaupr, 'Smin':smin, 'threshold':f1_max_threshold}
