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
    def __init__(self, embedding_matrix , hid_dim=512, n_layers=1, dropout=0.5):
        super().__init__()

        num_embeddings, embedding_dim = embedding_matrix.shape

        # self.embedding, num_embeddings, embedding_dim = create_emb_layer(weight_matrix, True)

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, _weight=torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False


        self.gru = nn.GRU(embedding_dim, hid_dim, n_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):

        embedded = self.embedding(inp)

        rnn_out, hidden = self.gru(embedded.float())
        # return self.rnn(self.embedding(inp))
        rnn_out = (rnn_out[:, :, :self.hid_dim] +
                rnn_out[:, :, self.hid_dim:])
        output = rnn_out[:,-1,:].squeeze(1)
        norm = output.norm(p=2, dim=1, keepdim=True)
        output = output.div(norm)
        return output
        

class ShapeDecoder(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.upsample1 = nn.ConvTranspose2d(embedding_size, embedding_size, 2, stride=1)
        self.upsample2 = nn.ConvTranspose2d(embedding_size, 256, 3, stride=1)
        self.upsample3 = nn.ConvTranspose2d(256, 128, 4, stride=2)
        self.upsample4 = nn.ConvTranspose2d(128, 6, 5, stride=3)
        self.l1 = nn.Linear(embedding_size,512)
        self.l2 = nn.Linear(512,976*6)
        
    def forward(self, inp):
        input = inp.reshape(inp.shape[0],inp.shape[1],1,1)
        out = self.upsample1(input)
        out = self.upsample2(out)
        out = self.upsample3(out)
        out = self.upsample4(out)
        out = out.reshape(-1,6,1024).permute(0,2,1)
        fc = self.l1(inp)
        fc = self.l2(fc)
        pc_fc = fc.reshape(-1, 976, 6)
        return torch.cat((out,pc_fc),dim=1)


