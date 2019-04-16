import sys
import os
import aer
import numpy as np

# reading training data
training_en = open('./training/hansards.36.2.e').read().splitlines()
training_fr = open('./training/hansards.36.2.f').read().splitlines()
# tokenize
vocab_tr_en = set()
vocab_tr_fr = set()
for i in range(len(training_en)):
    training_en[i] = training_en[i].split()
    for j in range(len(training_en[i])):
        vocab_tr_en.add(training_en[i][j])
for i in range(len(training_fr)):
    training_fr[i] = training_fr[i].split()
    for j in range(len(training_fr[i])):
        vocab_tr_fr.add(training_fr[i][j])
print(len(vocab_tr_en), len(vocab_tr_fr))

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
theta = dict.fromkeys(vocab_tr_en, dict.fromkeys(vocab_tr_fr, 1/len(vocab_tr_fr)))

# perform Expectation Maximization for IBM model 1
for iteration in range(iterations):
    # intialize counts
    count_p = dict.fromkeys(vocab_tr_en, dict.fromkeys(vocab_tr_fr, 0))
    count_w = dict.fromkeys(vocab_tr_en, 0)
    # E-Step
    for s_index in range(len(training_en)):
        for w_fr in training_fr[s_index]:
            # normalisation term
            Z = 0
            for w_en in training_en[s_index]:
                Z += theta[w_en][w_fr]
            for w_en in training_en[s_index]:
                c = theta[w_en][w_fr] / Z
                count_p[w_en][w_fr] += c
                count_w[w_en] += c
    for w_en in range(len(theta)):
        for w_fr in range(len(theta[w_en])):
            # M-Step
            theta[w_en][w_fr] = count_p[w_en][w_fr] / count_w[w_en]
    for s_index in range(len(validation_en)):
        for w_fr in validation_fr[s_index]:
            best_p = 0
            best_j = 0
            for i_en, w_en in enumerate(validation_en[s_index]):
                if theta[w_en][w_fr] > best_p:
                    best_p = theta[w_en][w_fr]
                    best_j = i_en
