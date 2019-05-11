from dataset import Vocabulary
from pre_data import preprocess
import os
import sys
import time
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from lstm_model import LSTMLM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_data = preprocess(open('./02-21.10way.clean', 'r', encoding='utf-8').read().splitlines())
print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
print('Number of sentences in training set: {}'.format(len(train_data)))
val_data = preprocess(open('./22.auto.clean', 'r', encoding='utf-8').read().splitlines())
print('Number of sentences in validation set: {}'.format(len(val_data)))
test_data = preprocess(open('./23.auto.clean', 'r', encoding='utf-8').read().splitlines())
print('Number of sentences in testing set: {}'.format(len(test_data)))
vocab = Vocabulary()
for sentence in train_data:
  for word in sentence:
      vocab.count_token(word)
vocab.build()   # build the dictionary
vocab_size = len(vocab.w2i)
print('Vocabulary size: {}'.format(vocab_size))
print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')

def prepare_example(example, vocab):
  """
  Map tokens to their IDs for 1 example
  """
  # vocab returns 0 if the word is not there
  x = [vocab.w2i.get(t, 0) for t in example[:-1]]
  x = torch.LongTensor([x])
  x = x.to(device)
  y = [vocab.w2i.get(t, 0) for t in example[1:]]
  y = torch.LongTensor([y])
  y = y.to(device)
  return x, y

def get_examples(data, shuffle=True, **kwargs):
  """Shuffle data set and return 1 example at a time (until nothing left)"""
  if shuffle:
    print("Shuffling training data...")
    random.shuffle(data)  # shuffle training data each epoch
  for example in data:
    yield example

def compute_perplexity(prediction, target):
  prediction = nn.functional.softmax(prediction, dim=2)
  perplexity = 0
  for i in range(prediction.shape[0]):
    for j in range(prediction.shape[1]):
      perplexity -= torch.log(prediction[i][j][int(target[i][j])])
  return perplexity

def train(config):
  # Print all configs to confirm parameter settings
  print_flags()

  # Initialize the model that we are going to use
  model = LSTMLM(vocabulary_size=vocab_size,
                  dropout=1-config.dropout_keep_prob,
                  lstm_num_hidden=config.lstm_num_hidden,
                  lstm_num_layers=config.lstm_num_layers,
                  device=device)
  model.to(device)

  # Setup the loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

  # Store some measures
  iteration = list()
  tmp_loss = list()
  train_loss = list()
  val_perp = list()
  iter_i = 0
  best_perp = 1e6

  while True:  # when we run out of examples, shuffle and continue
    for train_sen in get_examples(train_data, batch_size=1):

      # Only for time measurement of step through network
      t1 = time.time()
      iter_i += 1

      model.train()
      optimizer.zero_grad()

      inputs, targets = prepare_example(train_sen, vocab)

      pred = model(inputs)
      pred = pred.permute(0, 2, 1)
      loss = criterion(pred, targets)
      tmp_loss.append(loss.item())
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
      optimizer.step()

      # Just for time measurement
      t2 = time.time()
      examples_per_second = 1 / float(t2-t1)

      if iter_i % config.eval_every == 0:
        avg_loss = sum(tmp_loss) / len(tmp_loss)
        tmp_loss = list()
        model.eval()
        perp = list()
        length = list()
        for val_sen in val_data:
          val_input, val_target = prepare_example(val_sen, vocab)
          # forward pass
          # get the output from the neural network for input x
          with torch.no_grad():
            val_pred = model(val_input)
          tmp_per = float(compute_perplexity(val_pred, val_target))
          perp.append(tmp_per)
          length.append(val_target.shape[1])

        perplexity = np.exp(sum(perp) / sum(length))

        if perplexity < best_perp:
          best_perp = perplexity
          torch.save(model.state_dict(), "./models/lstm_best.pt")

        print("[{}] Train Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
              "Validation Perplexity = {:.2f}, Training Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), iter_i,
                config.train_steps, examples_per_second,
                perplexity, avg_loss
        ))
        iteration.append(iter_i)
        val_perp.append(perplexity)
        train_loss.append(avg_loss)

      if iter_i == config.train_steps:
        break
    
    if iter_i == config.train_steps:
      break
  print('Done training!')
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
  print('Testing...')
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
  model.load_state_dict(torch.load('./models/lstm_best.pt'))
  perp = list()
  length = list()
  for test_sen in test_data:
    test_input, test_target = prepare_example(test_sen, vocab)
    with torch.no_grad():
      test_pred = model(test_input)
    tmp_per = float(compute_perplexity(test_pred, test_target))
    perp.append(tmp_per)
    length.append(test_target.shape[1])
  test_perplexity = np.exp(sum(perp) / sum(length))
  print('Test Perplexity on the best model is: {:.2f}'.format(test_perplexity))
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
  with open('./result/lstm_test.txt', 'a') as file:
    file.write('Learning Rate = {}, Train Step = {}, '
               'Dropout = {}, LSTM Layers = {}, '
               'Hidden Size = {}, Test Perplexity = {:.2f}\n'.format(
                config.learning_rate, config.train_steps,
                1-config.dropout_keep_prob, config.lstm_num_layers,
                config.lstm_num_hidden, test_perplexity))
    file.close()
  fig, axs = plt.subplots(1, 2, figsize=(10,5))
  axs[0].plot(iteration, val_perp)
  axs[0].set_xlabel('Iteration')
  axs[0].set_ylabel('Validation Perplexity')
  axs[1].plot(iteration, train_loss)
  axs[1].set_xlabel('Iteration')
  axs[1].set_ylabel('Training Loss')
  fig.tight_layout()
  fig.savefig('./result/lstm_plot.eps', format='eps')
  print('Figure is saved.')
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
  plt.show()

def print_flags():
  """
  Prints all entries in config variable.
  """
  for key, value in vars(config).items():
    print(key + ' : ' + str(value))
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=100000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--eval_every', type=int, default=100, help='How often to print and evaluate training progress')

    config = parser.parse_args()

    # Train the model
    train(config)