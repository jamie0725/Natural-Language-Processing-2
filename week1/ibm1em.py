import sys
import os
from aer import read_naacl_alignments
import aer
import numpy as np
from decimal import *

path = 'validation/dev.wa.nonullalign'
# reading training data
training_en = open('./training/hansards.36.2.e').read().splitlines()
training_fr = open('./training/hansards.36.2.f').read().splitlines()
# tokenize
vocab_tr_en = set()
vocab_tr_en.add('NULLINDICATOR')
vocab_tr_fr = set()
for i in range(len(training_en)):
    training_en[i] = training_en[i].split()
    training_en[i] = ['NULLINDICATOR'] + training_en[i]
    for j in range(len(training_en[i])):
        vocab_tr_en.add(training_en[i][j])
for i in range(len(training_fr)):
    training_fr[i] = training_fr[i].split()
    for j in range(len(training_fr[i])):
        vocab_tr_fr.add(training_fr[i][j])
print('Training English Vocabulary:', len(vocab_tr_en))
print('Training French Vocabulary:', len(vocab_tr_fr))

# reading validation data
validation_en = open('./validation/dev.e').read().splitlines()
validation_fr = open('./validation/dev.f').read().splitlines()
# tokenize
for i in range(len(validation_en)):
    validation_en[i] = validation_en[i].split()
for i in range(len(validation_fr)):
    validation_fr[i] = validation_fr[i].split()

iterations = 10

# initialize theta
theta = dict.fromkeys(vocab_tr_fr, dict.fromkeys(vocab_tr_en, Decimal(1/len(vocab_tr_fr))))

# perform Expectation Maximization for IBM model 1
for iteration in range(iterations):
    # intialize counts
    count_p = dict.fromkeys(vocab_tr_fr, dict.fromkeys(vocab_tr_en, Decimal(0)))
    count_w = dict.fromkeys(vocab_tr_en, Decimal(0))
    # E-Step
    for s_index in range(len(training_en)):
        for w_fr in training_fr[s_index]:
            # normalisation term
            Z = Decimal(0)
            for w_en in training_en[s_index]:
                Z += Decimal(theta[w_fr][w_en])
            for w_en in training_en[s_index]:
                c = Decimal(theta[w_fr][w_en] / Z)
                count_p[w_fr][w_en] += Decimal(c)
                count_w[w_en] += Decimal(c)
    # print(len(theta), len(count_p), len(count_w))
    for w_fr in theta.keys():
        for w_en in theta[w_fr]:
            # M-Step
            theta[w_fr][w_en] = Decimal(count_p[w_fr][w_en] / count_w[w_en])
            # print('count_w[w_en]:', count_w[w_en])
            # print('count_p[w_fr][w_en]:', count_p[w_fr][w_en])
            # print('theta[w_fr][w_en]:', theta[w_fr][w_en])
            # assert theta[w_fr][w_en] <= 1.
    predictions = []
    print('2')
    for s_index in range(len(validation_en)):
        align = set()
        for i_fr, w_fr in enumerate(validation_fr[s_index]):
            best_p = 0
            best_j = 0
            for i_en, w_en in enumerate(validation_en[s_index]):
                if theta[w_en][w_fr] > best_p:
                    best_p = theta[w_en][w_fr]
                    best_j = i_en
            align.add((best_j, i_fr+1))
        predictions.append(align)
    gold_sets = read_naacl_alignments(path)
    metric = aer.AERSufficientStatistics()
    for gold, pred in zip(gold_sets, predictions):
        metric.update(sure=gold[0], probable=gold[1], predicted=pred)
    # AER
    print(metric.aer())