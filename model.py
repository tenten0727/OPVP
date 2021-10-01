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


class FFNN_v2(nn.Module):
    def __init__(self, num_columns, hidden_units=(256, 128, 64, 32), stock_embedding_dim=24):
        super(FFNN_v2, self).__init__()
        self.stock_embedding = nn.Embedding(127, stock_embedding_dim)
        self.flatten = nn.Flatten()
        layers = []
        for i in range(len(hidden_units)):
            if i == 0:
                layers.append(nn.Linear(num_columns+stock_embedding_dim, hidden_units[i]))
            else:
                layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            layers.append(nn.BatchNorm1d(hidden_units[i]))
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

class denoising_model(nn.Module):
    def __init__(self, num_columns):
        super(denoising_model,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(num_columns,256),
            # nn.BatchNorm1d(256),
            nn.SiLU(True),
            nn.Linear(256,128),
            # nn.BatchNorm1d(128),
            nn.SiLU(True),
        )
        
        self.decoder=nn.Sequential(
            nn.Linear(128,256),
            # nn.BatchNorm1d(256),
            nn.SiLU(True),
            nn.Linear(256, num_columns),
            # nn.BatchNorm1d(num_columns),
            nn.SiLU(True),
        )

        self.label_output = nn.Sequential(
            nn.Linear(num_columns, 256),
            nn.SiLU(True),
            nn.Linear(256, 64),
            nn.SiLU(True),
            nn.Linear(64, 1),
        )

    def forward(self, x, noise):
        x = x + noise
        x=self.encoder(x)
        x=self.decoder(x)
        output = self.label_output(x)
        return x, output
    
    def encode(self, x, noise):
        x = x + noise
        return self.encoder(x)

class DAE_FFNN(nn.Module):
    def __init__(self, num_columns, hidden_units=(256, 64), stock_embedding_dim=15):
        super(DAE_FFNN, self).__init__()
        self.stock_embedding = nn.Embedding(127, stock_embedding_dim)
        self.flatten = nn.Flatten()

        self.encoder=nn.Sequential(
            nn.Linear(num_columns,128),
            # nn.BatchNorm1d(256),
            nn.SiLU(True),
        )
        
        self.decoder=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, num_columns),
            # nn.BatchNorm1d(num_columns),
        )

        self.label_output = nn.Sequential(
            nn.Linear(num_columns, 256),
            nn.SiLU(True),
            nn.Linear(256, 1),
        )

        layers = []
        for i in range(len(hidden_units)):
            if i == 0:
                layers.append(nn.Linear(num_columns+stock_embedding_dim+128, hidden_units[i]))
            else:
                layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            layers.append(nn.SiLU(True))
        
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_units[-1], 1)

    def forward(self, x, stock, noise):
        emb = self.stock_embedding(stock)
        emb = self.flatten(emb)
        
        x_noise = x + noise

        x_encode = self.encoder(x_noise)
        x_decode = self.decoder(x_encode)
        label_out = self.label_output(x_decode)
        x = torch.cat((x, emb, x_encode), dim=1)
        x = self.hidden(x)
        out = self.out(x)
        return out, x_decode, label_out
    
    def forward_test(self, x, stock):
        emb = self.stock_embedding(stock)
        emb = self.flatten(emb)
        
        x_encode = self.encoder(x)
        x = torch.cat((x, emb, x_encode), dim=1)
        x = self.hidden(x)
        out = self.out(x)
        return out