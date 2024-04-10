import numpy as np
import pandas as pd

#### LOAD ONTOLOGY ####
def get_ancestors(ontology, term):
  list_of_terms = []
  list_of_terms.append(term)
  data = []
  
  while len(list_of_terms) > 0:
    new_term = list_of_terms.pop(0)

    if new_term not in ontology:
      break
    data.append(new_term)
    for parent_term in ontology[new_term]['parents']:
      if parent_term in ontology:
        list_of_terms.append(parent_term)
  
  return data

def generate_ontology(file, specific_space=False, name_specific_space=''):
  ontology = {}
  gene = {}
  flag = False
  with open(file) as f:
    for line in f.readlines():
      line = line.replace('\n','')
      if line == '[Term]':
        if 'id' in gene:
          ontology[gene['id']] = gene
        gene = {}
        gene['parents'], gene['alt_ids'] = [], []
        flag = True
        
      elif line == '[Typedef]':
        flag = False
      
      else:
        if not flag:
          continue
        items = line.split(': ')
        if items[0] == 'id':
          gene['id'] = items[1]
        elif items[0] == 'alt_id':
          gene['alt_ids'].append(items[1])
        elif items[0] == 'namespace':
          if specific_space:
            if name_specific_space == items[1]:
              gene['namespace'] = items[1]
            else:
              gene = {}
              flag = False
          else:
            gene['namespace'] = items[1]
        elif items[0] == 'is_a':
          gene['parents'].append(items[1].split(' ! ')[0])
        elif items[0] == 'name':
          gene['name'] = items[1]
        elif items[0] == 'is_obsolete':
          gene = {}
          flag = False
    
    key_list = list(ontology.keys())
    for key in key_list:
      ontology[key]['ancestors'] = get_ancestors(ontology, key)
      for alt_ids in ontology[key]['alt_ids']:
        ontology[alt_ids] = ontology[key]
    
    for key, value in ontology.items():
      if 'children' not in value:
        value['children'] = []
      for p_id in value['parents']:
        if p_id in ontology:
          if 'children' not in ontology[p_id]:
            ontology[p_id]['children'] = []
          ontology[p_id]['children'].append(key)
    
  return ontology

def get_and_print_children(ontology, term):
  children = {}
  if term in ontology:
    for i in ontology[term]['children']:
      children[i] = ontology[i]
      print(i, ontology[i]['name'])
  return children



#### GENERATE DATA BASED ON SLIDING WINDOW TECHNIQUE ####
def generate_label(indf):
    y = []
    label = []
    df = pd.read_csv(indf)
    for i in tqdm(range(len(df))):
        y.append(df.iloc[i, 2:])
  
    return df.columns.values[2:], np.array(y, dtype=int)



#### EMBEDDINGS EXTRACTION #### 
def save_numpy(path, file):
  with open(path, 'wb') as f:
    np.save(f, file)

