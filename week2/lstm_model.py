import torch.nn as nn


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
        return out, h_n, c_n
