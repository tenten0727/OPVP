import torch
import torch.nn as nn

class FFNN(nn.Module):
    def __init__(self, num_columns, hidden_units=(128, 64, 32), stock_embedding_dim=24):
        super(FFNN, self).__init__()
        self.stock_embedding = nn.Embedding(127, stock_embedding_dim)
        self.flatten = nn.Flatten()
        layers = []
        for i in range(len(hidden_units)):
            if i == 0:
                layers.append(nn.Linear(num_columns+stock_embedding_dim, hidden_units[i]))
            else:
                layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            layers.append(nn.SiLU(True))
        
        self.hidden = nn.Sequential(*layers)
        
        self.out = nn.Linear(hidden_units[-1], 1)
    def forward(self, x, stock):
        emb = self.stock_embedding(stock)
        emb = self.flatten(emb)
        x = torch.cat((x, emb), dim=1)
        x = self.hidden(x)
        out = self.out(x)
        return out