# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 11:14:15 2022

@author: Zhourun Wu
"""

import argparse
import numpy as np
import os
import scipy.io as sio
import networkx as nx
import pickle
from scipy import sparse

def get_proteins(protein_file):
    f = open(protein_file, 'r')
    f.readline()
    f_data = f.readlines()
    f.close()

    protein_name_dic = dict()

    for line in f_data:
        line_contents = line.split('\t')
        protein_name = line_contents[1]
        protein_externel_name = line_contents[0]
        if protein_name not in protein_name_dic:
            protein_name_dic[protein_name] = set([protein_externel_name])
        else:
            protein_name_dic[protein_name].add(protein_externel_name)
    return protein_name_dic

def get_ppi_proteins(ppi_file, org, data_path):
    f = open(ppi_file, 'r')
    f.readline()
    f_data = f.readlines()
    f.close()
    #f_data = f_data[:1000]
    ppi_graph = nx.Graph(name='ppi')
    
    for line in f_data:
        line_contents = line.split()
        protein_ext_name = line_contents[0]
        if not ppi_graph.has_node(protein_ext_name):
            ppi_graph.add_node(protein_ext_name)
        protein_ext_name = line_contents[1]
        if not ppi_graph.has_node(protein_ext_name):
            ppi_graph.add_node(protein_ext_name)
        
    print("number of ppi protein names is: ", ppi_graph.number_of_nodes())    
    return ppi_graph

def get_annots(ppi_graph, protein_name_dic, evidences, aspect, annot_data, train_date, valid_date, annotFile_start_line):    
    train_aspect_annot_dic = dict()
    valid_aspect_annot_dic = dict()
    test_aspect_annot_dic = dict()
    
    for i in range(annotFile_start_line, len(annot_data)):
        line_contents = annot_data[i].split('\t')
        go_id = line_contents[4]
        go_aspect = line_contents[8]
        protein_name = line_contents[2]
        evid = line_contents[6]
        annot_date = line_contents[13]
        if protein_name in protein_name_dic:
            is_pass = 1
            for epn in protein_name_dic[protein_name]:
                if not ppi_graph.has_node(epn):
                    is_pass = 0
                    break
            if go_aspect == aspect and evid in evidences and is_pass:
                if int(annot_date) <= int(train_date):
                    if go_id not in train_aspect_annot_dic:
                        train_aspect_annot_dic[go_id] = set()
                    train_aspect_annot_dic[go_id].add(protein_name)
                elif int(annot_date) <= int(valid_date):
                    if go_id not in valid_aspect_annot_dic:
                        valid_aspect_annot_dic[go_id] = set()
                    valid_aspect_annot_dic[go_id].add(protein_name)
                else:
                    if go_id not in test_aspect_annot_dic:
                        test_aspect_annot_dic[go_id] = set()
                    test_aspect_annot_dic[go_id].add(protein_name)
    annos = {}
    annos['train'] = train_aspect_annot_dic
    annos['valid'] = valid_aspect_annot_dic
    annos['test'] = test_aspect_annot_dic
    return annos
    

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
    
def get_proteinSet(anno_dic, good_terms):
    protein_set = set()
    for key in good_terms:
        protein_set = protein_set.union(anno_dic[key])
    return protein_set
        
def prune_inconsistTerms(annos):
    con_train_anno_dic = annos['train']
    con_valid_anno_dic = annos['valid']
    con_test_anno_dic = annos['test']
    
    for go_id in list(con_train_anno_dic):
        if go_id not in con_test_anno_dic:
            con_train_anno_dic.pop(go_id)
            
    for go_id in list(con_train_anno_dic):
        if go_id not in con_valid_anno_dic:
            con_train_anno_dic.pop(go_id)
            
    for go_id in list(con_valid_anno_dic):
        if go_id not in con_train_anno_dic:
            con_valid_anno_dic.pop(go_id)
            
    for go_id in list(con_valid_anno_dic):
        if go_id not in con_test_anno_dic:
            con_valid_anno_dic.pop(go_id)
            
    for go_id in list(con_test_anno_dic):
        if go_id not in con_train_anno_dic:
            con_test_anno_dic.pop(go_id)
            
    for go_id in list(con_test_anno_dic):
        if go_id not in con_valid_anno_dic:
            con_test_anno_dic.pop(go_id)
            
    con_anno_dic = {}
    con_anno_dic['train'] = con_train_anno_dic
    con_anno_dic['valid'] = con_valid_anno_dic
    con_anno_dic['test'] = con_test_anno_dic
    return con_anno_dic

def preprocess_annotations(ppi_graph, protein_name_dic, evidences,
                           annot_file, train_date, valid_date, data_path, org, annotFile_start_line):
    ppi_proteins = list(ppi_graph.nodes())
    num_proteins = len(ppi_proteins)
    

    f = open(annot_file, 'r')
    annot_data = f.readlines()
    f.close()
    
    Annot = {}
    Annot['GO'] = {}
    Annot['GO']['P'] = {}
    Annot['GO']['P']['train'] = []
    Annot['GO']['P']['valid'] = []
    Annot['GO']['P']['test'] = []
    Annot['GO']['F'] = {}
    Annot['GO']['F']['train'] = []
    Annot['GO']['F']['valid'] = []
    Annot['GO']['F']['test'] = []
    Annot['GO']['C'] = {}
    Annot['GO']['C']['train'] = []
    Annot['GO']['C']['valid'] = []
    Annot['GO']['C']['test'] = []
    Annot['indx'] = {}
    Annot['indx']['P'] = {}
    Annot['indx']['P']['train'] = []
    Annot['indx']['P']['valid'] = []
    Annot['indx']['P']['test'] = []
    Annot['indx']['F'] = {}
    Annot['indx']['F']['train'] = []
    Annot['indx']['F']['valid'] = []
    Annot['indx']['F']['test'] = []
    Annot['indx']['C'] = {}
    Annot['indx']['C']['train'] = []
    Annot['indx']['C']['valid'] = []
    Annot['indx']['C']['test'] = []
    Annot['labels'] = {}
    Annot['labels']['P'] = {}
    Annot['labels']['F'] = {}
    Annot['labels']['C'] = {}
    Annot['labels']['P']['terms'] = []
    Annot['labels']['F']['terms'] = []
    Annot['labels']['C']['terms'] = []
    
    
    for aspect in ['P', 'F', 'C']:
        annos = get_annots(ppi_graph, protein_name_dic, evidences, aspect,
                           annot_data, train_date, valid_date, annotFile_start_line)
        consist_annos = prune_inconsistTerms(annos)
        print("number of proteins is:", num_proteins)
        
        train_aspect_annot_dic = consist_annos['train']
        valid_aspect_annot_dic = consist_annos['valid']
        test_aspect_annot_dic = consist_annos['test']
        
        good_terms = []
        for go_id in list(train_aspect_annot_dic):
            go_term_proteins = len(train_aspect_annot_dic[go_id])
            protein_ratio = go_term_proteins / num_proteins
            if go_term_proteins >= 10 and protein_ratio <= 0.05:
                good_terms.append(go_id)
        print("number of annotations before preprocess is: ", aspect, ": ", len(good_terms))
        
        train_proteins = get_proteinSet(train_aspect_annot_dic, good_terms)
        
        p_valid_aspect_annot_dic = valid_aspect_annot_dic
        for go_id in valid_aspect_annot_dic:
            p_valid_aspect_annot_dic[go_id] = valid_aspect_annot_dic[go_id].difference(train_proteins)
        
        for go_id in list(valid_aspect_annot_dic):
            go_term_proteins = len(p_valid_aspect_annot_dic[go_id])
            protein_ratio = go_term_proteins / num_proteins
            if go_term_proteins < 5 or protein_ratio > 0.05:
                if go_id in good_terms:
                    good_terms.remove(go_id)
            
        train_proteins = get_proteinSet(train_aspect_annot_dic, good_terms)
        
        p_valid_aspect_annot_dic = valid_aspect_annot_dic
        for go_id in valid_aspect_annot_dic:
            p_valid_aspect_annot_dic[go_id] = valid_aspect_annot_dic[go_id].difference(train_proteins)
        
        valid_proteins = get_proteinSet(p_valid_aspect_annot_dic, good_terms)

        non_test_proteins = train_proteins.union(valid_proteins)
        test_proteins = set()
        p_test_aspect_annot_dic = test_aspect_annot_dic
        for go_id in test_aspect_annot_dic:
            p_test_aspect_annot_dic[go_id] = test_aspect_annot_dic[go_id].difference(non_test_proteins)
        
        for go_id in list(test_aspect_annot_dic):
            go_term_proteins = len(p_test_aspect_annot_dic[go_id])
            protein_ratio = go_term_proteins / num_proteins
            if go_term_proteins < 1 or protein_ratio > 0.05:
                if go_id in good_terms:
                    good_terms.remove(go_id)
                    
        train_proteins = get_proteinSet(train_aspect_annot_dic, good_terms)
        
        p_valid_aspect_annot_dic = valid_aspect_annot_dic
        for go_id in valid_aspect_annot_dic:
            p_valid_aspect_annot_dic[go_id] = valid_aspect_annot_dic[go_id].difference(train_proteins)
        valid_proteins = get_proteinSet(p_valid_aspect_annot_dic, good_terms)
        
        non_test_proteins = train_proteins.union(valid_proteins)
        
        p_test_aspect_annot_dic = test_aspect_annot_dic
        for go_id in test_aspect_annot_dic:
            p_test_aspect_annot_dic[go_id] = test_aspect_annot_dic[go_id].difference(non_test_proteins)
        test_proteins = get_proteinSet(p_test_aspect_annot_dic, good_terms)
        
        
        print("number of train proteins before preprocess is: ", aspect, ": ", len(train_proteins))
        print("number of valid proteins before preprocess is: ", aspect, ": ", len(valid_proteins))
        print("number of test proteins before preprocess is: ", aspect, ": ", len(test_proteins))
        
        num_aspect_annotations = len(good_terms)
        print("number of go terms is: ", num_aspect_annotations)
        
        print("number of train proteins after preprocess is: ", aspect, ": ", len(train_proteins))
        print("number of validation proteins after preprocess is: ", aspect, ": ", len(valid_proteins))
        print("number of test proteins after preprocess is: ", aspect, ": ", len(test_proteins))
        if len(test_proteins.intersection(train_proteins)) > 0:
            print('number intersection between train and test = ', len(test_proteins.intersection(train_proteins)))
        if len(test_proteins.intersection(valid_proteins)) > 0:
            print('number intersection between valid and test = ', len(test_proteins.intersection(valid_proteins)))
        if len(valid_proteins.intersection(train_proteins)) > 0:
            print('number intersection between train and valid = ', len(valid_proteins.intersection(train_proteins)))
        
        Annot['labels'][aspect]['terms'] = good_terms
        go_idx_dic = dict(zip(good_terms, range(len(good_terms))))
        num_aspect_annotations = len(good_terms)
        
        aa_train_set = set()
        aa_valid_set = set()
        aa_test_set = set()
        for go_id in good_terms:
            for protein_name in train_aspect_annot_dic[go_id]:
                aa_train_set.add(protein_name)
                protein_ext_names = protein_name_dic[protein_name]
                for x in protein_ext_names:
                    protein_id = ppi_proteins.index(x)
                    go_idx = go_idx_dic[go_id]
                    if protein_id not in Annot['indx'][aspect]['train']:
                        Annot['indx'][aspect]['train'].append(protein_id)
                        Annot['GO'][aspect]['train'].append(np.zeros(num_aspect_annotations, dtype = np.int).tolist())
                    protein_idx = Annot['indx'][aspect]['train'].index(protein_id)
                    Annot['GO'][aspect]['train'][protein_idx][go_idx] = 1
            for protein_name in p_valid_aspect_annot_dic[go_id]:
                aa_valid_set.add(protein_name)
                protein_ext_names = protein_name_dic[protein_name]
                for x in protein_ext_names:
                    protein_id = ppi_proteins.index(x)
                    go_idx = go_idx_dic[go_id]
                    if protein_id not in Annot['indx'][aspect]['valid']:
                        Annot['indx'][aspect]['valid'].append(protein_id)
                        Annot['GO'][aspect]['valid'].append(np.zeros(num_aspect_annotations, dtype = np.int).tolist())
                    protein_idx = Annot['indx'][aspect]['valid'].index(protein_id)
                    Annot['GO'][aspect]['valid'][protein_idx][go_idx] = 1
            for protein_name in p_test_aspect_annot_dic[go_id]:
                aa_test_set.add(protein_name)
                protein_ext_names = protein_name_dic[protein_name]
                for x in protein_ext_names:
                    protein_id = ppi_proteins.index(x)
                    go_idx = go_idx_dic[go_id]
                    if protein_id not in Annot['indx'][aspect]['test']:
                        Annot['indx'][aspect]['test'].append(protein_id)
                        Annot['GO'][aspect]['test'].append(np.zeros(num_aspect_annotations, dtype = np.int).tolist())
                    protein_idx = Annot['indx'][aspect]['test'].index(protein_id)
                    Annot['GO'][aspect]['test'][protein_idx][go_idx] = 1
        
        print("---number of train proteins after preprocess is: ", aspect, ": ", len(aa_train_set))
        print("---number of validation proteins after preprocess is: ", aspect, ": ", len(aa_valid_set))
        print("---number of test proteins after preprocess is: ", aspect, ": ", len(aa_test_set))
        
        save_file = os.path.join(data_path, org, org + '_annot.mat')
        sio.savemat(save_file, Annot)
        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', '--data_path', type = str, help = "the row data path")
    parser.add_argument('-af', '--annotation_file', type = str, help = "the input train annotation file")
    parser.add_argument('-pf', '--protein_file', type = str, help = "the input protein file")
    parser.add_argument('-ppif', '--ppi_file', type = str, help = "the input ppi file")
    parser.add_argument('-org', '--organism', type = str, help = "the organism")
    parser.add_argument('-stl', '--start_line', type = int, help = "the start line of Gene Ontology annotation file")
    
    margs = parser.parse_args()
    margs.annotation_file = os.path.join(margs.data_path, margs.organism, margs.annotation_file)
    margs.protein_file = os.path.join(margs.data_path, margs.organism, margs.protein_file)
    margs.ppi_file = os.path.join(margs.data_path, margs.organism, margs.ppi_file)
    
    if not os.path.exists(margs.data_path):
        print('data path not exist!')
        exit()
    protein_name_dic = get_proteins(margs.protein_file)
    ppi_graph = get_ppi_proteins(margs.ppi_file, margs.organism, margs.data_path)
    
    evidences = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC'])
    train_date = '20171231'
    valid_date = '20201231'
    preprocess_annotations(ppi_graph, protein_name_dic, evidences, 
                           margs.annotation_file, train_date, valid_date, margs.data_path, margs.organism, margs.start_line)
    