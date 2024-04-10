import ktrain
from ktrain import text
import os
import tensorflow as tf

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
def generate_data(df, subseq=100, overlap=0):
  X = []
  positions = []
  sequences = df.iloc[:, 1].values

  for i in tqdm(range(len(sequences))):

    len_seq = int(np.ceil(len(sequences[i]) / subseq))

    for idx in range(len_seq):
      if idx != len_seq - 1:
        X.append(' '.join(list(sequences[i][idx * subseq : (idx + 1) * subseq])))
      else:
        X.append(' '.join(list(sequences[i][idx * subseq : ])))
      positions.append(i)

    if overlap > 0:
      init = overlap
      while True:
        if init + subseq >= len(sequences[i]):
          break
        X.append(' '.join(list(sequences[i][init : init + subseq])))
        positions.append(i)
        init += subseq
  
  return X, positions



#### EMBEDDINGS EXTRACTION #### 
def save_numpy(path, file):
  with open(path, 'wb') as f:
    np.save(f, file)

def get_model(model_name):
  t = text.Transformer(model_name, maxlen=args.maxlen)
  trn = t.preprocess_train(X_train, y_train)
  model = t.get_classifier()
  learner = ktrain.get_learner(model, batch_size=args.batch, train_data=trn)
  predictor = ktrain.get_predictor(learner.model, preproc=t)
  full_model_name = f'weights/Esm2-full-model'
  learner.model.save_pretrained(full_model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  return TFAutoModel.from_pretrained(full_model_name), tokenizer, trn


def get_embeddings(X, model, tokenizer, file_path):
    batch_num = len(X)
    with NpyAppendArray(file_path) as npaa:
        for i in tqdm(range(batch_num)):
            with tf.device('/device:GPU:0'):
                outputs = model(X[i][0][:,0,:], X[i][0][:,1,:]) #input_ids, attention_mask
                outputs = outputs[0][:,0,:].numpy() # last_hidden_states, CLS token
                npaa.append(outputs)


#### COMBINE EMBEDDINGS ####
def protein_embedding(X, pos):
  n_X = []
  last_pos = pos[0]
  cur_emb = []
    
  for i in range(len(pos)):
    cur_pos = pos[i]
    if last_pos == cur_pos:
      cur_emb.append(X[i])
    else:
      n_X.append(np.mean(np.array(cur_emb), axis=0))
      last_pos = cur_pos
      cur_emb = [X[i]]

  n_X.append(np.mean(np.array(cur_emb), axis=0))
    
  return np.array(n_X)

