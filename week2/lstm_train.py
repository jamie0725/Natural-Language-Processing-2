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

def prepare_minibatch(mb, vocab):
  """
  Minibatch is a list of examples.
  This function converts words to IDs and returns
  torch tensors to be used as input/targets.
  """
  maxlen = max([len(sen) for sen in mb]) - 1
  # vocab returns 0 if the word is not there
  x = [pad([vocab.w2i.get(t, 0) for t in sen[:-1]], maxlen) for sen in mb]
  x = torch.LongTensor(x)
  x = x.to(device)
  y = [pad([vocab.w2i.get(t, 0) for t in sen[1:]], maxlen) for sen in mb]
  y = torch.LongTensor(y)
  y = y.to(device)
  return x, y

def get_minibatch(data, batch_size, shuffle=True):
  """Return minibatches, optional shuffling"""
  if shuffle:
    print("Shuffling training data")
    random.shuffle(data)  # shuffle training data each epoch
  batch = []
  # yield minibatches
  for example in data:
    batch.append(example)
    if len(batch) == batch_size:
      yield batch
      batch = []
  # in case there is something left
  if len(batch) > 0:
    yield batch

def pad(tokens, length, pad_value=1):
  """add padding 1s to a sequence to that it has the desired length"""
  return tokens + [pad_value] * (length - len(tokens))

def compute_perplexity(prediction, target):
  prediction = nn.functional.softmax(prediction, dim=2)
  perplexity = 0
  for i in range(prediction.shape[0]):
    for j in range(prediction.shape[1]):
      perplexity -= torch.log(prediction[i][j][int(target[i][j])])
  return float(perplexity)

def compute_match(prediction, target):
  match = 0
  pred = prediction.argmax(dim=2)
  match += (pred == target).sum().item()
  return int(match)

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
  criterion = nn.CrossEntropyLoss(ignore_index=1)
  optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

  # Store some measures
  iteration = list()
  tmp_loss = list()
  train_loss = list()
  val_perp = list()
  val_acc = list()
  iter_i = 0
  best_perp = 1e6

  while True:  # when we run out of examples, shuffle and continue
    for train_batch in get_minibatch(train_data, batch_size=config.batch_size):

      # Only for time measurement of step through network
      t1 = time.time()
      iter_i += 1

      model.train()
      optimizer.zero_grad()

      inputs, targets = prepare_minibatch(train_batch, vocab)

      print(inputs[0])
      print(targets[0])
      
      h_0 = torch.zeros(config.lstm_num_layers, inputs.shape[0], config.lstm_num_hidden).to(device)
      c_0 = torch.zeros(config.lstm_num_layers, inputs.shape[0], config.lstm_num_hidden).to(device)

      pred, _, _ = model(inputs, h_0, c_0)
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
        match = list()
        length = list()
        
        for val_sen in val_data:
          val_input, val_target = prepare_example(val_sen, vocab)
          h_0 = torch.zeros(config.lstm_num_layers, val_input.shape[0], config.lstm_num_hidden).to(device)
          c_0 = torch.zeros(config.lstm_num_layers, val_input.shape[0], config.lstm_num_hidden).to(device)

          with torch.no_grad():
            val_pred, _, _ = model(val_input, h_0, c_0)
          tmp_per = compute_perplexity(val_pred, val_target)
          tmp_match = compute_match(val_pred, val_target)
          perp.append(tmp_per)
          match.append(tmp_match)
          length.append(val_target.shape[1])

        perplexity = np.exp(sum(perp) / sum(length))
        accuracy = sum(match) / sum(length)

        if perplexity < best_perp:
          best_perp = perplexity
          torch.save(model.state_dict(), "./models/lstm_best.pt")

        print("[{}] Train Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
              "Validation Perplexity = {:.2f}, Training Loss = {:.3f}, "
              "Validation Accuracy = {:.2f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), iter_i,
                config.train_steps, examples_per_second,
                perplexity, avg_loss, accuracy
        ))
        iteration.append(iter_i)
        val_perp.append(perplexity)
        train_loss.append(avg_loss)
        val_acc.append(accuracy)

      if iter_i == config.train_steps:
        break
    
    if iter_i == config.train_steps:
      break
  
  print('Done training!')
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
  print('Testing...')
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
  
  model.load_state_dict(torch.load('./models/lstm_best.pt'))
  model.eval()
  perp = list()
  match = list()
  length = list()
  
  for test_sen in test_data:
    test_input, test_target = prepare_example(test_sen, vocab)
    h_0 = torch.zeros(config.lstm_num_layers, test_input.shape[0], config.lstm_num_hidden).to(device)
    c_0 = torch.zeros(config.lstm_num_layers, test_input.shape[0], config.lstm_num_hidden).to(device)
    with torch.no_grad():
      test_pred, _, _ = model(test_input, h_0, c_0)
    tmp_per = compute_perplexity(test_pred, test_target)
    tmp_match = compute_match(test_pred, test_target)
    perp.append(tmp_per)
    match.append(tmp_match)
    length.append(test_target.shape[1])
  
  test_perplexity = np.exp(sum(perp) / sum(length))
  test_accuracy = sum(match) / sum(length)
  
  print('Test Perplexity on the best model is: {:.2f}'.format(test_perplexity))
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
  with open('./result/lstm_test.txt', 'a') as file:
    file.write('Learning Rate = {}, Train Step = {}, '
               'Dropout = {}, LSTM Layers = {}, '
               'Hidden Size = {}, Test Perplexity = {:.2f}, '
               'Test Accuracy = {}'.format(
                config.learning_rate, config.train_steps,
                1-config.dropout_keep_prob, config.lstm_num_layers,
                config.lstm_num_hidden, test_perplexity, test_accuracy))
    file.close()
  
  generate = list()
  # Greedy decoding
  greedy = [vocab.w2i.get('SOS')]
  eos = vocab.w2i.get('EOS')
  greedy = torch.LongTensor([greedy]).unsqueeze(0).to(device)

  h_0 = torch.zeros(config.lstm_num_layers, greedy.shape[0], config.lstm_num_hidden).to(device)
  c_0 = torch.zeros(config.lstm_num_layers, greedy.shape[0], config.lstm_num_hidden).to(device)

  pred, h_n, c_n = model(greedy, h_0, c_0)

  pred = pred.argmax(dim=2)
  greedy = torch.cat((greedy, pred), dim=1)

  while greedy[-1][-1].item() != eos:
    pred, h_n, c_n = model(pred, h_n, c_n)
    pred = pred.argmax(dim=2)
    greedy = torch.cat((greedy, pred), dim=1)
  
  greedy = greedy.squeeze()
  g_sentence = [vocab.i2w(idx) for idx in greedy.tolist()]
  generate.append(g_sentence)

  for i in range(config.sample_size):
    sample = [vocab.w2i.get('SOS')]
    sample = torch.LongTensor([sample]).unsqueeze(0).to(device)
    h_0 = torch.zeros(config.lstm_num_layers, sample.shape[0], config.lstm_num_hidden).to(device)
    c_0 = torch.zeros(config.lstm_num_layers, sample.shape[0], config.lstm_num_hidden).to(device)

    pred, h_n, c_n = model(sample, h_0, c_0)
    pred = nn.functional.softmax(pred, dim=2)
    dist = torch.distributions.categorical.Categorical(pred)
    pred = dist.sample()
    sample = torch.cat((sample, pred), dim=1)

    while sample[-1][-1].item() != eos:
      pred, h_n, c_n = model(pred, h_n, c_n)
      pred = nn.functional.softmax(pred, dim=2)
      dist = torch.distributions.categorical.Categorical(pred)
      pred = dist.sample()
      sample = torch.cat((sample, pred), dim=1)
  
    sample = sample.squeeze()
    s_sentence = [vocab.i2w(idx) for idx in sample.tolist()]
    generate.append(s_sentence)

  with open('./result/lstm_test.txt', 'a') as file:
    for idx, sen in enumerate(generate):
      if idx == 0:
        file.write('Greedy: {}'.format(' '.join(sen)))
      else:
        file.write('Sampling {}: {}'.format(idx ,' '.join(sen)))
    file.close()

  t_loss = plt.figure(figsize = (6, 3))
  plt.plot(iteration, train_loss)
  plt.xlabel('Iteration')
  plt.ylabel('Training Loss')
  t_loss.savefig('./result/lstm_training_loss.eps', format='eps')
  v_perp = plt.figure(figsize = (6, 3))
  plt.plot(iteration, val_perp)
  plt.xlabel('Iteration')
  plt.ylabel('Validation Perplexity')
  v_perp.savefig('./result/lstm_validation_perplexity.eps', format='eps')
  v_acc = plt.figure(figsize = (6, 3))
  plt.plot(iteration, val_acc)
  plt.xlabel('Iteration')
  plt.ylabel('Validation Accuracy')
  v_acc.savefig('./result/lstm_validation_accuracy.eps', format='eps')
  print('Figures are saved.')
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')

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
    parser.add_argument('--batch_size', type=int, default=25, help='Batch size of the input')

    # Training params
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=30000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--eval_every', type=int, default=100, help='How often to print and evaluate training progress')
    parser.add_argument('--sample_size', type=int, default=10, help='Number of sampled sentences')

    config = parser.parse_args()

    # Train the model
    train(config)