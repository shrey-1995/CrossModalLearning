import torch
import torch.nn as nn
class Decoder(nn.Module):
    def __init__(self,embedding_matrix, output_dim, hid_dim, n_layers = 1, dropout = 0.3):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        num_embeddings, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, _weight=torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.rnn = nn.LSTM(embedding_dim, hid_dim, n_layers, batch_first=True)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        embedded = self.embedding(input)
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output)
        
        #prediction = [batch size, output dim]
        
        return prediction,hidden,cell
