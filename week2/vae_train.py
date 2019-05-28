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
from vae_model import LSTMLM
from vae_model import VAE

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

# print('train data head', train_data[0:5])
# print('train data tail', train_data[-5:])
# print('vocab 0,1', vocab.i2w[0], vocab.i2w[1]) #0=unk, 1=pad, 4=SOS, 6=EOS
# exit()

def prepare_example_numpy(example, vocab): # prepare_example keep it numpy for making batched copies in validation
  """
  Map tokens to their IDs for 1 example
  """
  # vocab returns 0 if the word is not there
  x = [vocab.w2i.get(t, 0) for t in example[:-1]]
  # x = torch.LongTensor([x])
  # x = x.to(device)
  y = [vocab.w2i.get(t, 0) for t in example[1:]]
  # y = torch.LongTensor([y])
  # y = y.to(device)
  return x, y

def prepare_example(example, vocab): # prepare_example keep it numpy for making batched copies in validation
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

  # sort the minibatch for padding removing before biLSTM
  mb.sort(reverse=True, key=len)
  maxlen = len(mb[0])-1

  # maxlen = max([len(sen) for sen in mb]) - 1

  # vocab returns 0 if the word is not there
  x = [pad([vocab.w2i.get(t, 0) for t in sen[:-1]], maxlen) for sen in mb]
  x = torch.LongTensor(x)
  x = x.to(device)
  y = [pad([vocab.w2i.get(t, 0) for t in sen[1:]], maxlen) for sen in mb]
  y = torch.LongTensor(y)
  y = y.to(device)

  # also return the unpadded lengths of all sents in a batch 
  # (for the pack_padded function in VAE model forward, which removes the the padding for faster computation)
  lengths_in_batch = [len(sen)-1 for sen in mb]

  return x, y, lengths_in_batch 
  # x is setnences from first word to secLast word (in vocab index)
  # y is setnences from sec word to last word (in vocab index)

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

def compute_perplexity(prediction, target): # the negative log-likelihood term in perplexity(for RNNLM only)
  prediction = nn.functional.softmax(prediction, dim=2)
  perplexity = 0
  for i in range(prediction.shape[0]): # batch shape  (=1 when validating)
    for j in range(prediction.shape[1]): # sentence length, ie 1 word/1 timestamp for each loop
      perplexity -= torch.log(prediction[i][j][int(target[i][j])])
  return float(perplexity)
'''
def compute_match(prediction, target):
  match = 0
  pred = prediction.argmax(dim=2)
  match += (pred == target).sum().item()
  return int(match)
'''
def compute_match_vae(prediction, target):
  match = 0
  pred = prediction.argmax(dim=1)
  match += (pred == target).sum().item()
  return int(match)


def train(config):
  # Print all configs to confirm parameter settings
  print_flags()

  # Initialize the model that we are going to use
  # model = LSTMLM(vocabulary_size=vocab_size,
  model = VAE(vocabulary_size=vocab_size,
                  dropout=1-config.dropout_keep_prob,
                  lstm_num_hidden=config.lstm_num_hidden,
                  lstm_num_layers=config.lstm_num_layers,
                  lstm_num_direction=config.lstm_num_direction,
                  num_latent=config.num_latent,
                  device=device)
  
  # Setup the loss and optimizer
  criterion = nn.CrossEntropyLoss(ignore_index=1, reduction='sum')
  optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

  # Store some measures
  iteration = list()
  tmp_loss = list()
  train_loss = list()
  val_perp = list()
  val_acc = list()
  val_elbo = list()
  iter_i = 0
  best_perp = 1e6
  

  while True:  # when we run out of examples, shuffle and continue
    for train_batch in get_minibatch(train_data, batch_size=config.batch_size):

      # Only for time measurement of step through network
      t1 = time.time()
      iter_i += 1

      model.train()
      optimizer.zero_grad()

      inputs, targets, lengths_in_batch = prepare_minibatch(train_batch, vocab)
      
      # zeros in dim = (num_layer*num_direction * batch * lstm_hidden_size)
      # we have bidrectional single layer LSTM
      h_0 = torch.zeros(config.lstm_num_layers*config.lstm_num_direction, inputs.shape[0], config.lstm_num_hidden).to(device)
      c_0 = torch.zeros(config.lstm_num_layers*config.lstm_num_direction, inputs.shape[0], config.lstm_num_hidden).to(device)

      # pred, _, _ = model(inputs, h_0, c_0)
      decoder_output, KL_loss= model(inputs, h_0, c_0, lengths_in_batch, config.importance_sampling_size)


      reconstruction_loss = 0.0

      for k in range(config.importance_sampling_size):
        # the first argument for criterion, ie, crossEntrooy must be (batch, classes(ie vocab size), sent_length), so we need to permute the last two dimension of decoder_output (batch, sent_length, vocab_classes)
        # decoder_output[k] =decoder_output[k].permute(0, 2, 1) doesnt work 
        reconstruction_loss += criterion(decoder_output[k].permute(0, 2, 1), targets)

      # get the mean of the k samples of z 
      reconstruction_loss = reconstruction_loss/config.importance_sampling_size 
      KL_loss = KL_loss/config.importance_sampling_size 


      # print('At iter', iter_i, ', rc_loss=', reconstruction_loss.item(), ' KL_loss = ', KL_loss.item())

      total_loss= (reconstruction_loss+ KL_loss)/config.batch_size
      tmp_loss.append(total_loss.item())
      total_loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
      optimizer.step()


      if iter_i % config.eval_every == 0:
        # print('Evaluating with validation at iteration ', iter_i, '...')
        model.eval()

        ppl_total = 0.0
        validation_elbo_loss = 0.0
        validation_lengths=list()
        match=list()


        with torch.no_grad():
          # computing ppl, match, and accuracy
          for validation_th, val_sen in enumerate(val_data): # too large too slow lets stick with first 1000/1700 first
            val_input, val_target = prepare_example(val_sen, vocab)
            

            # zeros in dim = (num_layer*num_direction, batch=config.importance_sampling_size,  lstm_hidden_size)
            h_0 = torch.zeros(config.lstm_num_layers*config.lstm_num_direction, config.importance_sampling_size, config.lstm_num_hidden).to(device)
            c_0 = torch.zeros(config.lstm_num_layers*config.lstm_num_direction, config.importance_sampling_size, config.lstm_num_hidden).to(device)


            # append the sent length of this particular validation example
            validation_lengths.append(val_input.size(1))

            # feed into models 
            decoder_output, KL_loss_validation= model(val_input, h_0, c_0, [val_input.size(1)], config.importance_sampling_size)
            

            # decoder_output.size() = (k, batchsize=1, val_input.size(1)(ie sent_length), vocabsize)
            # prediction.size() = (k, sent_len, vocabsize)
            # prediction_mean.size() = (sent_len, vocabsize), ie averaged over k samples (and squeezed)
            prediction = nn.functional.softmax(torch.squeeze(decoder_output, dim=1), dim=2)
            prediction_mean = torch.mean(prediction, 0) #averaged over k 


            ppl_per_example = 0.0
            for j in range(prediction.shape[1]): # sentence length, ie 1 word/1 timestamp for each loop
              ppl_per_example -= torch.log(prediction_mean[j][int(val_target[0][j])]) # 0 as the target is the same for the k samples

            ppl_total+= ppl_per_example

            # if validation_th % 300 == 0:
            #   print('    ppl_per_example at the ', validation_th, ' th validation case = ', ppl_per_example)

            tmp_match = compute_match_vae(prediction_mean, val_target)
            match.append(tmp_match)

            # calculate validation elbo
            # decoder_output.size() = (k, batchsize=1, val_input.size(1)(ie sent_length), vocabsize)
            # the first argument for criterion, ie, crossEntrooy must be (batch, classes(ie vocab size), sent_length), so we need to permute the last two dimension of decoder_output  to get (k, batchsize=1, vocab_classes, sent_length)
            # then we loop over k to get (1, vocab_classes, sent_len)
            decoder_output_validation = decoder_output.permute(0, 1, 3, 2)

            reconstruction_loss=0

            for k in range(config.importance_sampling_size):
              reconstruction_loss += criterion(decoder_output_validation[k], val_target)


            validation_elbo_loss+= (reconstruction_loss+ KL_loss_validation)/config.importance_sampling_size


        ppl_total = torch.exp(ppl_total/sum(validation_lengths))
        # print('ppl_total for iteration ', iter_i, ' =  ', ppl_total)

        accuracy = sum(match) / sum(validation_lengths)
        # print('accuracy for iteration ', iter_i, ' =  ', accuracy)

        avg_loss = sum(tmp_loss) / len(tmp_loss) # loss of the previous iterations (up the after last eval)
        tmp_loss = list() # reinitialize to zero
        validation_elbo_loss = validation_elbo_loss/len(val_data)


        if ppl_total < best_perp:
          best_perp = ppl_total
          torch.save(model.state_dict(), "./models/vae_best.pt")

          # Instead of rewriting the same file, we can have new ones:
          # model_saved_name = datetime.now().strftime("%Y-%m-%d_%H%M") + './models/vae_best.pt'
          # torch.save(model.state_dict(), model_saved_name)

        print("[{}] Train Step {:04d}/{:04d}, "
              "Validation Perplexity = {:.4f}, Validation loss ={:.4f}, Training Loss = {:.4f}, "
              "Validation Accuracy = {:.4f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), iter_i,
                config.train_steps,
                ppl_total, validation_elbo_loss, avg_loss, accuracy
        ))
        iteration.append(iter_i)
        val_perp.append(ppl_total.item())
        train_loss.append(avg_loss)
        val_acc.append(accuracy)
        val_elbo.append(validation_elbo_loss.item())

        # np.save('./np_saved_results/train_loss.npy', train_loss + ['till_iter_'+str(iter_i)])
        # np.save('./np_saved_results/val_perp.npy', val_perp+['till_iter_'+str(iter_i)])
        # np.save('./np_saved_results/val_acc.npy', val_acc+['till_iter_'+str(iter_i)])
        # np.save('./np_saved_results/val_elbo.npy', val_elbo+['till_iter_'+str(iter_i)])



      if iter_i == config.train_steps:
        np.save('./np_saved_results/train_loss.npy', train_loss + ['till_iter_'+str(iter_i)])
        np.save('./np_saved_results/val_perp.npy', val_perp+['till_iter_'+str(iter_i)])
        np.save('./np_saved_results/val_acc.npy', val_acc+['till_iter_'+str(iter_i)])
        np.save('./np_saved_results/val_elbo.npy', val_elbo+['till_iter_'+str(iter_i)])
        break
    
    if iter_i == config.train_steps:
      break
  
  
  print('Done training!')
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')

  print('Testing...')
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
  model.load_state_dict(torch.load('./models/vae_best.pt'))
  model.eval()

  ppl_total = 0.0
  validation_elbo_loss = 0.0
  validation_lengths=list()
  match=list()


  with torch.no_grad():
    # computing ppl, match, and accuracy
    for validation_th, val_sen in enumerate(test_data): #too large too slow lets stick with first 1000/1700 first
      val_input, val_target = prepare_example(val_sen, vocab)
      

      # zeros in dim = (num_layer*num_direction, batch=config.importance_sampling_size,  lstm_hidden_size)
      h_0 = torch.zeros(config.lstm_num_layers*config.lstm_num_direction, config.importance_sampling_size, config.lstm_num_hidden).to(device)
      c_0 = torch.zeros(config.lstm_num_layers*config.lstm_num_direction, config.importance_sampling_size, config.lstm_num_hidden).to(device)


      # append the sent length of this particular validation example
      validation_lengths.append(val_input.size(1))

      # feed into models 
      decoder_output, KL_loss_validation= model(val_input, h_0, c_0, [val_input.size(1)], config.importance_sampling_size)
      

      # decoder_output.size() = (k, batchsize=1, val_input.size(1)(ie sent_length), vocabsize)
      # prediction.size() = (k, sent_len, vocabsize)
      # prediction_mean.size() = (sent_len, vocabsize), ie averaged over k samples (and squeezed)
      prediction = nn.functional.softmax(torch.squeeze(decoder_output, dim=1), dim=2)
      prediction_mean = torch.mean(prediction, 0) #averaged over k 


      ppl_per_example = 0.0
      for j in range(prediction.shape[1]): # sentence length, ie 1 word/1 timestamp for each loop
        ppl_per_example -= torch.log(prediction_mean[j][int(val_target[0][j])])#0 as the target is the same for the k samples

      ppl_total+= ppl_per_example

      tmp_match = compute_match_vae(prediction_mean, val_target)
      match.append(tmp_match)

      # calculate validation elbo
      # decoder_output.size() = (k, batchsize=1, val_input.size(1)(ie sent_length), vocabsize)
      # the first argument for criterion, ie, crossEntrooy must be (batch, classes(ie vocab size), sent_length), so we need to permute the last two dimension of decoder_output  to get (k, batchsize=1, vocab_classes, sent_length)
      # then we loop over k to get (1, vocab_classes, sent_len)
      decoder_output_validation = decoder_output.permute(0, 1, 3, 2)

      reconstruction_loss=0

      for k in range(config.importance_sampling_size):
        reconstruction_loss += criterion(decoder_output_validation[k], val_target)


      validation_elbo_loss+= (reconstruction_loss+ KL_loss_validation)/config.importance_sampling_size


  ppl_total = torch.exp(ppl_total/sum(validation_lengths))

  accuracy = sum(match) / sum(validation_lengths)

  validation_elbo_loss = validation_elbo_loss/len(val_data)


  print('Test Perplexity on the best model is: {:.3f}'.format(ppl_total))
  print('Test ELBO on the best model is: {:.3f}'.format(validation_elbo_loss))
  print('Test accuracy on the best model is: {:.3f}'.format(accuracy))
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
  with open('./result/vae_test.txt', 'a') as file:
    file.write('Learning Rate = {}, Train Step = {}, '
               'Dropout = {}, LSTM Layers = {}, '
               'Hidden Size = {}, Test Perplexity = {:.3f}, Test ELBO =  {:.3f},'
               'Test Accuracy = {}\n'.format(
                config.learning_rate, config.train_steps,
                1-config.dropout_keep_prob, config.lstm_num_layers,
                config.lstm_num_hidden, ppl_total, validation_elbo_loss, accuracy))
    file.close()


  print('Sampling...')
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')

  model.load_state_dict(torch.load('./models/vae_best.pt'))
  
  with torch.no_grad():
    sentences = model.sample(config.sample_size, vocab)

  with open('./result/vae_test.txt', 'a') as file:
    for idx, sen in enumerate(sentences):
        file.write('Sampling {}: {}\n'.format(idx ,' '.join(sen)))




  print('Done sampling!')
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')


  t_loss = plt.figure(figsize = (6, 4))
  plt.plot(iteration, train_loss)
  plt.xlabel('Iteration')
  plt.ylabel('Training Loss')
  t_loss.tight_layout()
  t_loss.savefig('./result/vae_training_loss.eps', format='eps')
  
  v_perp = plt.figure(figsize = (6, 4))
  plt.plot(iteration, val_perp)
  plt.xlabel('Iteration')
  plt.ylabel('Validation Perplexity')
  v_perp.tight_layout()
  v_perp.savefig('./result/vae_validation_perplexity.eps', format='eps')

  v_acc = plt.figure(figsize = (6, 4))
  plt.plot(iteration, val_acc)
  plt.xlabel('Iteration')
  plt.ylabel('Validation Accuracy')
  v_acc.tight_layout()
  v_acc.savefig('./result/vae_validation_accuracy.eps', format='eps')


  v_elbo = plt.figure(figsize = (6, 4))
  plt.plot(iteration, val_elbo)
  plt.xlabel('Iteration')
  plt.ylabel('Validation ELBO')
  v_elbo.tight_layout()
  v_elbo.savefig('./result/vae_validation_elbo.eps', format='eps')
  print('Figures are saved.')
  print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
  


  return 0

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
    parser.add_argument('--lstm_num_layers', type=int, default=1, help='Number of LSTM layers in the model')
    parser.add_argument('--lstm_num_direction', type=int, default=2, help='Number of LSTM direction, 2 for bidrectional')
    parser.add_argument('--num_latent', type=int, default=64, help='latent size of the input')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size of the input')

    # Training params
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=3000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--eval_every', type=int, default=100, help='How often to print and evaluate training progress')
    parser.add_argument('--sample_size', type=int, default=10, help='Number of sampled sentences')

    # size of k in z_{nk}, ie how many z to we want to average for ppl 
    parser.add_argument('--importance_sampling_size', type=int, default=2, help='Number of z sampled per validation example for importances sampling')

    config = parser.parse_args()

    # Train the model
    train(config)