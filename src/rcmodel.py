[1mdiff --git a/src/model.py b/src/model.py[m
[1mindex b686f90..4766748 100644[m
[1m--- a/src/model.py[m
[1m+++ b/src/model.py[m
[36m@@ -1,6 +1,7 @@[m
 import torch[m
 import torch.nn as nn[m
 import torch.nn.functional as F[m
[32m+[m[32mimport math[m
 [m
 class LSTMClassifier(nn.Module):[m
     """LSTM model for 3-class price prediction"""[m
[36m@@ -113,3 +114,64 @@[m [mclass TCNClassifier(nn.Module):[m
         [m
         return x[m
     [m
[32m+[m
[32m+[m[32mclass PositionalEncoding(nn.Module):[m
[32m+[m[32m    """Positional encoding for Transformer"""[m
[32m+[m[41m    [m
[32m+[m[32m    def __init__(self, d_model, max_len=5000, dropout=0.1):[m
[32m+[m[32m        super().__init__()[m
[32m+[m[32m        self.dropout = nn.Dropout(p=dropout)[m
[32m+[m[41m        [m
[32m+[m[32m        pe = torch.zeros(max_len, d_model)[m
[32m+[m[32m        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)[m
[32m+[m[32m        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))[m
[32m+[m[41m        [m
[32m+[m[32m        pe[:, 0::2] = torch.sin(position * div_term)[m
[32m+[m[32m        pe[:, 1::2] = torch.cos(position * div_term)[m
[32m+[m[32m        pe = pe.unsqueeze(0)[m
[32m+[m[41m        [m
[32m+[m[32m        self.register_buffer('pe', pe)[m
[32m+[m[41m    [m
[32m+[m[32m    def forward(self, x):[m
[32m+[m[32m        x = x + self.pe[:, :x.size(1), :][m
[32m+[m[32m        return self.dropout(x)[m
[32m+[m
[32m+[m
[32m+[m[32mclass TransformerClassifier(nn.Module):[m
[32m+[m[32m    def __init__(self, input_size, d_model=128, nhead=8, num_layers=4,[m[41m [m
[32m+[m[32m                 dim_feedforward=512, dropout=0.1):[m
[32m+[m[32m        super().__init__()[m
[32m+[m[41m        [m
[32m+[m[32m        self.d_model = d_model[m
[32m+[m[32m        self.input_size = input_size[m
[32m+[m[41m        [m
[32m+[m[32m        self.input_projection = nn.Linear(input_size, d_model)[m
[32m+[m[32m        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)[m
[32m+[m[41m        [m
[32m+[m[32m        encoder_layer = nn.TransformerEncoderLayer([m
[32m+[m[32m            d_model=d_model,[m
[32m+[m[32m            nhead=nhead,[m
[32m+[m[32m            dim_feedforward=dim_feedforward,[m
[32m+[m[32m            dropout=dropout,[m
[32m+[m[32m            batch_first=True[m
[32m+[m[32m        )[m
[32m+[m[41m        [m
[32m+[m[32m        self.transformer_encoder = nn.TransformerEncoder([m
[32m+[m[32m            encoder_layer,[m
[32m+[m[32m            num_layers=num_layers[m
[32m+[m[32m        )[m
[32m+[m[41m        [m
[32m+[m[32m        self.fc1 = nn.Linear(d_model, 64)[m
[32m+[m[32m        self.fc2 = nn.Linear(64, 3)[m
[32m+[m[32m        self.dropout = nn.Dropout(dropout)[m
[32m+[m[32m        self.relu = nn.ReLU()[m
[32m+[m[41m        [m
[32m+[m[32m    def forward(self, x):[m
[32m+[m[32m        x = self.input_projection(x)[m
[32m+[m[32m        x = self.pos_encoder(x)[m
[32m+[m[32m        x = self.transformer_encoder(x)[m
[32m+[m[32m        x = x[:, -1, :][m
[32m+[m[32m        x = self.dropout(x)[m
[32m+[m[32m        x = self.relu(self.fc1(x))[m
[32m+[m[32m        x = self.fc2(x)[m
[32m+[m[32m        return x[m
