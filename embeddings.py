from text_extraction_torch import get_text_embedding
import torch
import torch.nn as nn

import numpy as np

import random

'''

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True'''


# https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
# def create_emb_layer(weights_matrix=embedding_matrix, non_trainable=True):
#     num_embeddings, embedding_dim = weights_matrix.shape
#     emb_layer = nn.Embedding(num_embeddings, embedding_dim)
#     emb_layer.load_state_dict({'weight': weights_matrix})
#     if non_trainable:
#         emb_layer.weight.requires_grad = False
#
#     return emb_layer, num_embeddings, embedding_dim

# https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
# Encoder
class Encoder(nn.Module):
    def __init__(self, weight_matrix , hid_dim=512, n_layers=2, dropout=0.5):
        super().__init__()

        num_embeddings, embedding_dim = weight_matrix.shape

        # self.embedding, num_embeddings, embedding_dim = create_emb_layer(weight_matrix, True)

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, _weight=torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False


        self.rnn = nn.LSTM(embedding_dim, hid_dim, n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp=encoder_input_sentences):

        embedded = self.dropout(self.embedding(inp))

        outputs, (hidden, cell) = self.rnn(embedded.float())
        # return self.rnn(self.embedding(inp))

        return outputs









