import torch.nn as nn
import torch
import torch.nn.functional as F
import math

##########################################
#                                                                                                        #
#                                   MLP Model                                                     #
#                                                                                                        #
##########################################

class MLP(nn.Module):
    """
    Network architecture based on MLP to train on SEED/SEED5 dataset
    """

    def __init__(self, n_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features=310, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=n_classes)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.dropout1 = nn.Dropout(p=0.8)
        self.dropout2 = nn.Dropout(p=0.8)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, x):
        feats = self.bn1(self.fc1(x))
        feats = self.relu1(feats)
        feats = self.dropout1(feats)
        feats = self.bn2(self.fc2(feats))
        feats = self.relu2(feats)
        feats = self.dropout2(feats)
        logits = self.fc3(feats)
        return feats, logits

##########################################
#                                                                                                        #
#                                   LSTM Model                                                   #
#                                                                                                        #
##########################################

class LSTM(nn.Module):
    def __init__(self, n_classes,
                 input_dim=310,
                 hidden_dim=64,
                 num_layers=1,
                 dropout=0.3):

        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim,
                            hidden_dim,
                            num_layers,
                            batch_first=True
                            )
        self.dropout = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_classes)  # Directly connect to output

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM output
        feats = self.bn(lstm_out[:, -1, :])  # Take last time step
        feats = self.dropout(feats)
        logits = self.fc(feats)
        return feats, logits  # Removed unnecessary intermediate features



##########################################
#                                                                                                        #
#                                   Transformer Model                                         #
#                                                                                                        #
##########################################


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class Transformer(nn.Module):
    def __init__(self, n_classes,
                 input_dim=310,
                 d_model=128,
                 num_heads=1,
                 num_layers=1,
                 dropout=0.3
                 ):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # Feature projection
        self.bn1 = nn.BatchNorm1d(d_model)  # BatchNorm after embedding
        self.pos_encoder = PositionalEncoding(d_model)  # Add positional encoding

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.bn2 = nn.BatchNorm1d(d_model)  # BatchNorm before classification
        self.fc = nn.Linear(d_model, n_classes)  # Classification layer

    def forward(self, x):
        x = self.embedding(x)  # Map input to lower-dim space
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)  # Apply BatchNorm across feature dim
        x = self.pos_encoder(x)  # Inject positional information
        x = self.transformer(x)  # Transformer encoder
        x = x[:, -1, :]  # Take last token (like LSTM last hidden state)
        # x = x.mean(dim=1)  # Take last token (like LSTM last hidden state)
        x = self.bn2(x)  # Apply BatchNorm before classification
        logits = self.fc(x)
        return x, logits