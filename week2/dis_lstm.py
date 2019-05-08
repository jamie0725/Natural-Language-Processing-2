from pre_data import preprocess
import torch.nn as nn
import torch

train_data = preprocess(open('./02-21.10way.clean', 'r', encoding='utf-8').read().splitlines())
print('Number of sentences in training set: {}'.format(len(train_data)))
print('Example sentence in training set:', train_data[0])
val_data = preprocess(open('./22.auto.clean', 'r', encoding='utf-8').read().splitlines())
print('\nNumber of sentences in validation set: {}'.format(len(val_data)))
print('Example sentence in validation set:', val_data[0])
test_data = preprocess(open('./23.auto.clean', 'r', encoding='utf-8').read().splitlines())
print('\nNumber of sentences in testing set: {}'.format(len(test_data)))
print('Example sentence in testing set:', test_data[0])