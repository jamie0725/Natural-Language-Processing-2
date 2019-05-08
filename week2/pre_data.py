import os
import sys
from nltk.tree import Tree

def preprocess(data):
    '''
    Input {list}: a tree corpus
    Output {list}: a sentence corpus starting with 'SOS' and ending with 'EOS'
    Example usage:
        from pre_data import preprocess
        train_data = preprocess(open('./02-21.10way.clean', 'r', encoding='utf-8').read().splitlines())
    '''
    processed = list()
    for tree in data:
        tree = Tree.fromstring(tree)
        sen = tree.leaves()
        sen = ['SOS'] + sen + ['EOS']
        processed.append(sen)
    return processed
