# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 11:14:15 2022

@author: Zhourun Wu
"""

import argparse
import numpy as np
import os
import scipy.io as sio
import pandas as pd
import pickle
import gzip
import networkx as nx

def get_proteins(protein_file):
    if protein_file.endswith('.gz'):
        f = gzip.open(protein_file, 'rt')
    elif protein_file.endswith('.txt'):
        f = open(protein_file, 'r')
    f.readline()
    f_data = f.readlines()
    f.close()

    protein_name2extname_dic = dict()
    protein_extname2name_dic = dict()

    for line in f_data:
        line_contents = line.split('\t')
        protein_name = line_contents[1]
        protein_externel_name = line_contents[0]
        if protein_name not in protein_name2extname_dic:
            protein_name2extname_dic[protein_name] = set([protein_externel_name])
        else:
            protein_name2extname_dic[protein_name].add(protein_externel_name)
        if protein_externel_name not in protein_extname2name_dic:
            protein_extname2name_dic[protein_externel_name] = protein_name
    return protein_name2extname_dic, protein_extname2name_dic

def get_ppi_proteins(ppi_file, org, data_path):
    if ppi_file.endswith('.gz'):
        f = gzip.open(ppi_file, 'rt')
    elif ppi_file.endswith('.txt'):
        f = open(ppi_file, 'r')
    f.readline()
    f_data = f.readlines()
    #f_data = f_data[:1000]
    f.close()
    
    g = nx.Graph()
    
    predict_PPIs = set()
    all_PPIs = set()
    for line in f_data:
        line_contents = line.split()
        protein_ext_name1 = line_contents[0]
        if not g.has_node(protein_ext_name1):
            g.add_node(protein_ext_name1)
        protein_ext_name2 = line_contents[1]
        if not g.has_node(protein_ext_name2):
            g.add_node(protein_ext_name2)
        if int(line_contents[-2]) > 0:
            flag = False
            for j in range(2,8):
                if int(line_contents[j]) > 0:
                    flag = True
                    break
            if not flag:
                predict_PPIs.add((protein_ext_name1, protein_ext_name2))
        all_PPIs.add((protein_ext_name1, protein_ext_name2))
    print('ratio of textmining PPIs = ', len(predict_PPIs) / len(all_PPIs))
    ppi_proteins_l = list(g.nodes())
    
    return ppi_proteins_l
    

def save_annotations(protein_aspect_annotation_file, protein_aspect_annotation_dic):    
    f = open(protein_aspect_annotation_file, 'w')
    for protein_name in protein_aspect_annotation_dic:
        f.write(protein_name + '\t')
        for i in protein_aspect_annotation_dic[protein_name]:
            f.write(str(i) + '\t')
        f.write('\n')
    f.close()
    
def save_dict(dictionary_file, dictionary):
    f = open(dictionary_file, 'w')
    for key in dictionary:
        f.write(str(key) + '\t' + str(dictionary[key]) + '\n')
    f.close()

def process_sub_loc(x):
    if str(x) == 'nan':
        return []
    x = x[22:-1]
    # check if exists "Note="
    pos = x.find("Note=")
    if pos != -1:
        x = x[:(pos-2)]
    temp = [t.strip() for t in x.split(".")]
    temp = [t.split(";")[0] for t in temp]
    temp = [t.split("{")[0].strip() for t in temp]
    temp = [x for x in temp if '}' not in x and x != '']
    return temp

def process_domain(x):
    if str(x) == 'nan':
        return []
    temp = [t.strip() for t in x[:-1].split(";")]
    return temp

def preprocess_features(ppi_proteins, protein_name2extname_dic, protein_extname2name_dic,
                        data_path, org, uniprot_file):
    ppi_proteins = list(ppi_proteins)
    num_proteins = len(ppi_proteins)
    
    protein_id_dic = dict(zip(ppi_proteins,range(num_proteins)))
    
    Annot = pd.DataFrame()
    
    
    uniprot = pd.read_table(uniprot_file)
    if org == 'human':
        Annot['Gene names  (primary )'] = uniprot['Gene names  (primary )']
    else:
        Annot['Gene names  (primary )'] = uniprot['Gene Names (primary)']
    
    Annot['Sub_cell_loc'] = uniprot['Subcellular location [CC]'].apply(process_sub_loc)
    for i in range(len(Annot['Sub_cell_loc'])):
        if str(Annot.loc[i, 'Sub_cell_loc']) == 'nan':
            Annot.at[i, 'Sub_cell_loc'] = []
    
    if org == 'human':
        Annot['Cross-reference (STRING)'] = uniprot['Cross-reference (STRING)']
    else:
        Annot['Cross-reference (STRING)'] = uniprot['STRING']
    items = [item for sublist in Annot['Sub_cell_loc'] for item in sublist]
    items = np.unique(items)
    sub_mapping = dict(zip(list(items),range(len(items))))
    sub_encoding = [[0]*len(items) for i in range(len(ppi_proteins))]
    for i,row in Annot.iterrows():
        p_extn = ''
        if not pd.isna(row['Cross-reference (STRING)']):
            p_extn = str(row['Cross-reference (STRING)']).strip(';')
            if p_extn in protein_id_dic:
                for loc in row['Sub_cell_loc']:
                    sub_encoding[protein_id_dic[p_extn]][sub_mapping[loc]] = 1
    
    sub_encoding = np.array(sub_encoding)
    if org == 'human':
        Annot['protein-domain'] = uniprot['Cross-reference (Pfam)'].apply(process_domain)
    else:
        Annot['protein-domain'] = uniprot['Pfam'].apply(process_domain)
    for i in range(len(Annot['protein-domain'])):
        if str(Annot.loc[i, 'protein-domain']) == 'nan':
            Annot.at[i, 'protein-domain'] = []
    items = [item for sublist in Annot['protein-domain'] for item in sublist]
    unique_elements, counts_elements = np.unique(items, return_counts=True)
    items = unique_elements[np.where(counts_elements > 5)]
    pro_mapping = dict(zip(list(items),range(len(items))))
    pro_encoding = [[0]*len(items) for i in range(len(ppi_proteins))]
    
    for i,row in Annot.iterrows():
        p_extn = ''
        if not pd.isna(row['Cross-reference (STRING)']):
            p_extn = str(row['Cross-reference (STRING)']).strip(';')
            if p_extn in protein_id_dic:
                for fam in row['protein-domain']:
                    if fam in pro_mapping:
                        pro_encoding[protein_id_dic[p_extn]][pro_mapping[fam]] = 1

    pro_encoding = np.array(pro_encoding)
    
    #### wirte files
    print("write files...")
    print('sub_encoding shpae=',sub_encoding.shape)
    print('domain_encoding shpae=',pro_encoding.shape)
    g2g_features = np.column_stack((sub_encoding, pro_encoding))
    print('g2g_features shpae=',g2g_features.shape)
    s_file = os.path.join(data_path, org, "features.npy")
    with open(s_file, 'wb') as f:
        pickle.dump(g2g_features, f)
    
    s_file = os.path.join(data_path, org, org + '_proteins.txt')
    with open(s_file, 'wb') as f:
        pickle.dump(ppi_proteins, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', '--data_path', type = str, help = "the data_path")
    parser.add_argument('-pf', '--protein_file', type = str, help = "the input protein file")
    parser.add_argument('-ppif', '--ppi_file', type = str, help = "the input ppi file")
    parser.add_argument('-org', '--organism', type = str, help = "the data_path")
    parser.add_argument('-uniprot', '--uniprot_file', type = str, help = "uniprot_file")
    
    margs = parser.parse_args()
    if not os.path.exists(margs.data_path):
        os.mkdir(margs.data_path)
    margs.protein_file = os.path.join(margs.data_path, margs.organism, margs.protein_file)
    margs.ppi_file = os.path.join(margs.data_path, margs.organism, margs.ppi_file)
    margs.uniprot_file = os.path.join(margs.data_path, margs.organism, margs.uniprot_file)
    
    protein_name2extname_dic, protein_extname2name_dic = get_proteins(margs.protein_file)
    ppi_graph = get_ppi_proteins(margs.ppi_file, margs.organism, margs.data_path)
    
        
    preprocess_features(ppi_graph, protein_name2extname_dic, protein_extname2name_dic,
                           margs.data_path, margs.organism, margs.uniprot_file)
    
