import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMLM(nn.Module):

    def __init__(self, vocabulary_size, dropout,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(LSTMLM, self).__init__()

        # embedding
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                      embedding_dim=lstm_num_hidden,
                                      padding_idx=1)

        # layers
        self.model = nn.LSTM(input_size=lstm_num_hidden, 
                             hidden_size=lstm_num_hidden,
                             num_layers=lstm_num_layers, 
                             bias=True,
                             dropout=dropout,
                             batch_first=True)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)

        self.to(device)


    def forward(self, x, h, c):
        embed = self.embedding(x)
        model, (h_n, c_n) = self.model(embed, (h, c))
        out = self.linear(model)
        print('out size', out.size())
        return out, h_n, c_n


class VAE(nn.Module):
    def __init__(self, vocabulary_size, dropout,
                 lstm_num_hidden=128, lstm_num_layers=1, lstm_num_direction=2, num_latent=32 , device='cuda:0'):

        super(VAE, self).__init__()

        self.lstm_num_hidden = lstm_num_hidden

        #encoder 
        self.biLSTM_encoder = nn.LSTM(input_size=lstm_num_hidden, 
                             hidden_size=lstm_num_hidden,
                             num_layers=lstm_num_layers, 
                             bias=True,
                             dropout=dropout,
                             batch_first=True, 
                             bidirectional=True)
        #embedding 
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                      embedding_dim=lstm_num_hidden,
                                      padding_idx=1)        

        #latent size of z
        self.num_latent = num_latent


        #mean
        self.mu=nn.Linear(lstm_num_hidden*lstm_num_direction*lstm_num_layers, num_latent) # *2 as it's bidirectional 

        #for variance
        self.logvar=nn.Linear(lstm_num_hidden*lstm_num_direction*lstm_num_layers, num_latent) # *2 as it's bidirectional 

        #to do, add this in according to proj description 
        self.softplus = nn.Softplus()


        #latent to decoder 
        self.latent2decoder = nn.Linear(num_latent, lstm_num_hidden) #single layer single direction LSTM

        #decoder
        self.LSTM_decoder = nn.LSTM(input_size=lstm_num_hidden, 
                             hidden_size=lstm_num_hidden,
                             num_layers=lstm_num_layers, 
                             bias=True,
                             dropout=dropout,
                             batch_first=True, 
                             bidirectional=False) #unidirectional

        self.LSTM_output = nn.Linear(lstm_num_hidden, vocabulary_size)




        self.device=device
        self.to(device)




    def forward(self, x, h_0, c_0, lengths_in_batch):
        ''' 
        debugging log: just in case something goes wrong in the future, here's a list of things already checked
        #print('input(batch) size', x.size())
        #= batch * sent_len 

        #h_N, (h_t, c_t) = self.biLSTM_encoder(embedded, (h_0, c_0))

        #print('h_N', h_N.size()) equals all h-outputs of each timestamp 
        # = batch * sent_len * (lstm_num_hidden * #direction * #layer)
        # = batch * sent_len * (128*2)

        #print('h_t', h_t.size()) only the output of last timestamp t
        # = (#direction * #layer) * batch * lstm_num_hidden


        #checking that the last of h_N matches h_t
        #print('h_N 0th batch, last h, head', h_N[0, -1, 0:5])
        #print('h_t 0th batch, last h, head', h_t[:, 0, 0:5])

        #print('h_N', h_N.size())
        #print('h_N last ', torch.squeeze(h_N[:, -1, :]).size())

        #trying to pick the final feature h_t as the output of the biLSTM_encoder encoder
        #encoded = torch.squeeze(h_N[:, -1, :])


        #checking if pack_padded works
        #print('x', x)
        #print('lengths_in_batch',lengths_in_batch)
        #print('pack_padded_sequence(embedded)',pack_padded_sequence(embedded, lengths=lengths_in_batch,batch_first=True, enforce_sorted=False))


        #checking how to convert packed output from lstm to what we needed 
        #print('x.size()',x.size())
        #print('lengths_in_batch',lengths_in_batch)
        #print('h_N_packed.data.shape ', h_N_packed.data.shape)

        #checking unpacked output 
        #print('h_N_unpacked size', h_N_unpacked.size())
        #print('h_t_packed size', h_t_packed.size())
        #print('h_t_packed -1 ', h_t_packed[-1].size())

        #checking if encoder output works
        #encoder_output = torch.cat((h_t_packed[0], h_t_packed[1]),dim=1)
        #print('encoder_outputsize', encoder_output.size())
        #print('encoder_output', encoder_output)

        #checking dimensions of z     
        #print('z rand', z)
        #print('z rand size', z.size())
        #print('mu shape', mu.shape)

        #checking decode
        #print('z.size()',z.size())
        #print('z',z)
        
        #decoder_input = self.latent2decoder(z)
        #print('decoder_input.size()',decoder_input.size())
        #print('decoder_input',decoder_input)

        #checking the final output matches the desired output dim
        print('decoder_output size and batch max length, lens_in_batch',decoder_output.size(), x.size(1), lengths_in_batch)
        print('x',x)
        '''



        #embed input 
        embedded = self.embedding(x)

        #remove the paddings for faster computation
        packed_embedded = pack_padded_sequence(embedded, lengths=lengths_in_batch,batch_first=True, enforce_sorted=False)


        #feed pad_removed input into encoder 
        #h_N_packed, (h_t_packed, c_t_packed) = self.biLSTM_encoder(packed_embedded, (h_0, c_0))
        _, (h_t_packed, _) = self.biLSTM_encoder(packed_embedded, (h_0, c_0))    

        '''
        In case this is needed in future: to convert h_N_unpacked(padding removed) back to h_N_unpacked(with padding - padded position has [0,...0] as h_i); dim of h_N_unpacked = batch * sent * (lstm_num_hidden * #direction * #layer)

        h_N_unpacked, lengths_in_batch_just_the_same = pad_packed_sequence(h_N_packed, batch_first=True)
        '''

        #The h_t_packed has weird dim = (num_layer * num direction) * batch * lstm_hidden = 2 * batch * 128
        #have to concat the 2 directions of 128 hidden states back to 256
        #s.t. its dim now =  batch * 256
        encoder_output = torch.cat((h_t_packed[0], h_t_packed[1]),dim=1)

        #mean 
        mu = self.mu(encoder_output)

        #log variance
        logvar= self.logvar(encoder_output)

        #std
        std= torch.exp(.5*logvar)

        #introduce the epsilon randomness (actually default of requires grad is already false, anyway ...)
        z = torch.randn((mu.shape), requires_grad=False).to(self.device)

        #compute z
        z = z*std + mu  

        #compute the KL loss 
        KL_loss = 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar)

        

        #map the latent dimensions of z back to the lstm_num_hidden dimensions        
        decoder_input = self.latent2decoder(z)

        #unsqueeze is for adding one dim of 1 to fit the input constraint of LSTM:
        #Inputs: input, (h_0, c_0); h_0 of shape (num_layers * num_directions, batch, hidden_size)
        decoder_hidden_init = decoder_input.unsqueeze(0)

        #use this if want to init cell state with z as well:
        #decoder_cell_init = decoder_input.clone().unsqueeze(0)
        
        #Use this instead if take init cell state as empty: (which is the first attempt)
        decoder_cell_init = torch.zeros(1, x.size(0), self.lstm_num_hidden).to(self.device)




        #feed this new z to the LSTM decoder to get all the hidden states 
        #Am I right to feeed z to initial hidden states (instead of cell states?)
        h_N_packed, (_, _)  = self.LSTM_decoder(packed_embedded, (decoder_hidden_init, decoder_cell_init))

        #h_N_unpacked contains hidden states output of all timesteps 
        #unused return value is the the lengths_in_batch: h_N_unpacked, lengths_in_batch = pad_packed_sequence(h_N_packed, batch_first=True)
        h_N_unpacked, _ = pad_packed_sequence(h_N_packed, batch_first=True)


        #decoder output is the fully-connected layer from num_hidden to vocab size, 
        #Then in train.py we will use nn.CrossEntropy as the softmax to calculate the loss from this decoder_ouput 
        decoder_output = self.LSTM_output(h_N_unpacked)


        return decoder_output, KL_loss
