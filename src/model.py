import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTMClassifier(nn.Module):
    """LSTM model for 3-class price prediction"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 3)  # 3 classes: down, no change, up
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        out, (h_n, c_n) = self.lstm(x)
        
        # Take last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out
  
class CausalConv1d(nn.Module):
    """1D causal convolution (only looks at past)"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=self.padding, dilation=dilation)
        
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding != 0 else x


class TCNBlock(nn.Module):
    """TCN residual block with dilated causal convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        if self.downsample:
            residual = self.downsample(residual)
            
        return self.relu(out + residual)


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network for time series classification"""
    
    def __init__(self, input_size, num_channels=[64, 64, 64], kernel_size=3, dropout=0.3):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8...
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 3)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # → (batch, features, seq_len)
        
        x = self.network(x)
        x = x[:, :, -1]  # Take last time step
        x = self.fc(x)
        
        return x
    

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.input_size = input_size
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
