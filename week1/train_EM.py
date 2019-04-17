import sys
import os
import numpy as np
import operator
from aer import read_naacl_alignments
import aer

def preprocess(s):
    a = " ".join(s.split())
    return a.split(" ")

def init_theta(en_train, fr_train):
    theta_0 = {}
    count_f_e = {}
    count_e = {}
    
    N= len(en_train)
    for n in range(N):
        
        if (n % 1000) == 0:
            print(n)
            
        en_n = en_train[n]
        cleaned_en_n = ["ILOVEVINCENT"] + preprocess(en_n) 
        fr_n = fr_train[n]
        cleaned_fr_n = preprocess(fr_n)
        
        for j_en_word in cleaned_en_n:
            if j_en_word not in theta_0:
                theta_0[j_en_word] = {}
            if j_en_word not in count_e:
                count_e[j_en_word] = 0
            for i_fr_word in cleaned_fr_n:
                if i_fr_word not in theta_0[j_en_word]:
                    theta_0[j_en_word][i_fr_word] = 0
                    
                pair_f_e = []
                pair_f_e.append(i_fr_word)
                pair_f_e.append(j_en_word)
                pair_f_e = tuple(pair_f_e)
                if pair_f_e not in count_f_e:
                    count_f_e[pair_f_e] = 0
                    
    for en_word in theta_0.keys():
        count = len(theta_0[en_word].keys())
        for fr_word in theta_0[en_word].keys():
            theta_0[en_word][fr_word] = 1/count
            
    return theta_0, count_f_e, count_e

def train_EM(en_train, fr_train, theta_0, count_f_e_0, count_e_0, K=100):
    path = 'validation/dev.wa.nonullalign'
    
    N = len(en_train)
    theta = theta_0
    theta_new = theta.copy()
    
    for k in range(K):
        print(k)
        count_f_e = count_f_e_0.copy()
        count_e = count_e_0.copy()
        for n in range(N):
            en_n = en_train[n]
            cleaned_en_n = ["ILOVEVINCENT"] + preprocess(en_n)
            fr_n = fr_train[n]
            cleaned_fr_n = preprocess(fr_n)
            for i in range(len(cleaned_fr_n)):
                Z = 0
                for j in range(len(cleaned_en_n)):
                    Z += theta[cleaned_en_n[j]][cleaned_fr_n[i]]
                for j in range(len(cleaned_en_n)):
                    c = theta[cleaned_en_n[j]][cleaned_fr_n[i]] / Z
                    f_e = (cleaned_fr_n[i], cleaned_en_n[j])
                    count_f_e[f_e] += c
                    count_e[cleaned_en_n[j]] += c
        for f_e in count_f_e.keys():
            fr = f_e[0]
            en = f_e[1]
            theta_new[en][fr] = count_f_e[f_e] / count_e[en]
        
        theta = theta_new.copy()
        
        predictions = []
        for s_index in range(len(en_val)):
            align = set()
            for i_fr, w_fr in enumerate(preprocess(en_val[s_index])):
                best_p = 0
                best_j = 0
                for i_en, w_en in enumerate(preprocess(en_val[s_index])):
                    if w_en in theta_new and w_fr in theta_new[w_en] and theta_new[w_en][w_fr] > best_p:
                        best_p = theta_new[w_en][w_fr]
                        best_j = i_en
                align.add((best_j, i_fr+1))
            predictions.append(align)
        gold_sets = read_naacl_alignments(path)
        metric = aer.AERSufficientStatistics()
        for gold, pred in zip(gold_sets, predictions):
            metric.update(sure=gold[0], probable=gold[1], predicted=pred)
        print(metric.aer())
        
    return theta_new

if __name__ == "__main__":
    # reading training data
    en_train = open('./training/hansards.36.2.e').read().splitlines()
    fr_train = open('./training/hansards.36.2.f').read().splitlines()

    # reading validation data
    en_val = open('./validation/dev.e').read().splitlines()
    fr_val = open('./validation/dev.f').read().splitlines()

    theta_0, count_f_e_0, count_e_0 = init_theta(en_train, fr_train)

    theta = train_EM(en_train, fr_train, theta_0, count_f_e_0, count_e_0, K=10)