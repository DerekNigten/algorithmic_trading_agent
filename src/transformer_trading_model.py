"""
Transformer Model for Financial Time Series Prediction
IE 421: Agentic AI in Financial Markets
Author: Naina's Team

This module implements a Transformer-based model for predicting short-term price movements
using IEX DEEP market data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class FinancialTimeSeriesDataset(Dataset):
    """
    Dataset class for financial time series data from IEX DEEP.
    
    This handles the creation of sequences for the Transformer model.
    """
    
    def __init__(self, data: pd.DataFrame, sequence_length: int, 
                 feature_columns: List[str], target_column: str,
                 prediction_horizon: int = 1):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame with financial data
            sequence_length: Length of input sequences (lookback window)
            feature_columns: List of feature column names
            target_column: Name of target column (e.g., 'future_return')
            prediction_horizon: Steps ahead to predict (default: 1)
        """
        self.data = data
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.prediction_horizon = prediction_horizon
        
        # Prepare features and targets
        self.features = data[feature_columns].values
        self.targets = data[target_column].values
        
        # Calculate valid indices (accounting for sequence length and prediction horizon)
        self.valid_indices = range(
            sequence_length, 
            len(data) - prediction_horizon + 1
        )
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Get a single sequence and its target.
        
        Returns:
            tuple: (sequence, target) where sequence is [seq_len, n_features]
        """
        actual_idx = self.valid_indices[idx]
        
        # Get sequence of features
        sequence = self.features[actual_idx - self.sequence_length:actual_idx]
        
        # Get target (future value)
        target = self.targets[actual_idx + self.prediction_horizon - 1]
        
        return (
            torch.FloatTensor(sequence),
            torch.FloatTensor([target])
        )


class PositionalEncoding(nn.Module):
    """
    Positional encoding to inject information about token positions.
    Uses sinusoidal functions as in the original Transformer paper.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the model (embedding dimension)
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerFinancialModel(nn.Module):
    """
    Transformer model for financial time series prediction.
    
    Architecture:
    1. Input embedding layer (projects features to d_model dimensions)
    2. Positional encoding
    3. Transformer encoder layers
    4. Output projection layer
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 100
    ):
        """
        Initialize the Transformer model.
        
        Args:
            n_features: Number of input features
            d_model: Dimension of model embeddings
            n_heads: Number of attention heads
            n_layers: Number of Transformer encoder layers
            d_ff: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Input embedding: project input features to d_model dimensions
        self.input_embedding = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True  # Expect input as [batch, seq, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 1)
        
    def forward(self, x, src_mask=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]
            src_mask: Optional attention mask
            
        Returns:
            predictions: Tensor of shape [batch_size, 1]
        """
        # Input embedding
        x = self.input_embedding(x) * np.sqrt(self.d_model)
        
        # Transpose for positional encoding [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # Back to [batch_size, seq_len, d_model]
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(x, src_mask)
        
        # Use the last time step's output for prediction
        last_output = transformer_out[:, -1, :]
        
        # Output layers
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        predictions = self.fc2(x)
        
        return predictions


class TransformerTrader:
    """
    Main class for training and using the Transformer model for trading.
    """
    
    def __init__(
        self,
        sequence_length: int = 50,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: str = None
    ):
        """
        Initialize the TransformerTrader.
        
        Args:
            sequence_length: Length of input sequences
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of Transformer layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            device: Device to use ('cuda' or 'cpu')
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model hyperparameters
        self.model_params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'd_ff': d_ff,
            'dropout': dropout,
            'max_seq_length': sequence_length
        }
        
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        print(f"Using device: {self.device}")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = 'future_return',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data for training, validation, and testing.
        
        Args:
            df: DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Name of target column
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            
        Returns:
            train_loader, val_loader, test_loader
        """
        self.feature_columns = feature_columns
        
        # Normalize features
        df_normalized = df.copy()
        df_normalized[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        
        # Split data chronologically (important for time series!)
        n = len(df_normalized)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = df_normalized[:train_end]
        val_data = df_normalized[train_end:val_end]
        test_data = df_normalized[val_end:]
        
        print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Create datasets
        train_dataset = FinancialTimeSeriesDataset(
            train_data, self.sequence_length, feature_columns, target_column
        )
        val_dataset = FinancialTimeSeriesDataset(
            val_data, self.sequence_length, feature_columns, target_column
        )
        test_dataset = FinancialTimeSeriesDataset(
            test_data, self.sequence_length, feature_columns, target_column
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def build_model(self, n_features: int):
        """
        Build the Transformer model.
        
        Args:
            n_features: Number of input features
        """
        self.model = TransformerFinancialModel(
            n_features=n_features,
            **self.model_params
        ).to(self.device)
        
        print(f"\nModel Architecture:")
        print(f"  - Input features: {n_features}")
        print(f"  - Model dimension: {self.model_params['d_model']}")
        print(f"  - Attention heads: {self.model_params['n_heads']}")
        print(f"  - Encoder layers: {self.model_params['n_layers']}")
        print(f"  - Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for sequences, targets in train_loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(sequences)
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(sequences)
                loss = criterion(predictions, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        patience: int = 10
    ):
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Maximum number of epochs
            patience: Early stopping patience
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print("\nStarting training...")
        print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Status'}")
        print("-" * 50)
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            status = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_transformer_model.pth')
                status = "✓ Saved"
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"{epoch+1:<8} {train_loss:<12.6f} {val_loss:<12.6f} {status}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_transformer_model.pth'))
        print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
        
        return train_losses, val_losses
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """
        Make predictions on new sequences.
        
        Args:
            sequences: Array of shape [n_samples, seq_len, n_features]
            
        Returns:
            predictions: Array of shape [n_samples]
        """
        self.model.eval()
        
        with torch.no_grad():
            sequences_tensor = torch.FloatTensor(sequences).to(self.device)
            predictions = self.model(sequences_tensor)
            
        return predictions.cpu().numpy().flatten()
    
    def backtest(
        self,
        test_loader: DataLoader,
        decision_threshold: float = 0.0
    ) -> dict:
        """
        Backtest the model on test data.
        
        Args:
            test_loader: Test data loader
            decision_threshold: Threshold for buy/sell decisions
            
        Returns:
            Dictionary with backtest results
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                predictions = self.model(sequences)
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(targets.numpy().flatten())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        mse = np.mean((all_predictions - all_targets) ** 2)
        mae = np.mean(np.abs(all_predictions - all_targets))
        
        # Trading signals: 1 (buy), 0 (hold), -1 (sell)
        signals = np.where(all_predictions > decision_threshold, 1,
                          np.where(all_predictions < -decision_threshold, -1, 0))
        
        # Calculate accuracy (direction prediction)
        correct_direction = np.sum(np.sign(all_predictions) == np.sign(all_targets))
        directional_accuracy = correct_direction / len(all_targets)
        
        # Simple P&L calculation (assuming we trade based on signals)
        returns = signals[:-1] * all_targets[1:]  # Shift by 1 for realistic trading
        cumulative_returns = np.cumsum(returns)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        results = {
            'mse': mse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'total_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'n_trades': np.sum(signals != 0),
            'predictions': all_predictions,
            'targets': all_targets,
            'signals': signals
        }
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Mean Squared Error:      {mse:.6f}")
        print(f"Mean Absolute Error:     {mae:.6f}")
        print(f"Directional Accuracy:    {directional_accuracy:.2%}")
        print(f"Total Return:            {results['total_return']:.4f}")
        print(f"Sharpe Ratio:            {sharpe_ratio:.4f}")
        print(f"Number of Trades:        {results['n_trades']}")
        print("="*50)
        
        return results
    
    def save_model(self, filepath: str):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_params': self.model_params,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, n_features: int):
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model_params = checkpoint['model_params']
        self.scaler = checkpoint['scaler']
        self.feature_columns = checkpoint['feature_columns']
        self.sequence_length = checkpoint['sequence_length']
        
        self.build_model(n_features)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")


def create_features_from_iex_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from raw IEX DEEP data.
    
    This function should be customized based on your specific IEX data structure.
    
    Args:
        df: DataFrame with IEX DEEP data
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Example features (customize based on your IEX data):
    
    # 1. Returns
    df['return'] = df['price'].pct_change()
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    
    # 2. Volume features
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # 3. Price features
    df['price_ma_5'] = df['price'].rolling(window=5).mean()
    df['price_ma_20'] = df['price'].rolling(window=20).mean()
    df['price_ma_ratio'] = df['price_ma_5'] / df['price_ma_20']
    
    # 4. Volatility
    df['volatility_5'] = df['return'].rolling(window=5).std()
    df['volatility_20'] = df['return'].rolling(window=20).std()
    
    # 5. Order book imbalance (if available in IEX DEEP)
    if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
        df['order_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
    
    # 6. Spread (if available)
    if 'bid_price' in df.columns and 'ask_price' in df.columns:
        df['spread'] = df['ask_price'] - df['bid_price']
        df['spread_pct'] = df['spread'] / df['price']
    
    # 7. Technical indicators
    # RSI
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['price'].ewm(span=12, adjust=False).mean()
    exp2 = df['price'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # 8. Target: future return (what we want to predict)
    df['future_return'] = df['return'].shift(-1)  # Next period's return
    
    # Drop NaN values
    df = df.dropna()
    
    return df


# Example usage
if __name__ == "__main__":
    print("Transformer Model for Financial Trading")
    print("=" * 50)
    print("\nThis script provides a complete implementation of a Transformer model")
    print("for predicting short-term price movements from IEX DEEP data.")
    print("\nTo use this model:")
    print("1. Load your IEX data into a pandas DataFrame")
    print("2. Use create_features_from_iex_data() to engineer features")
    print("3. Initialize TransformerTrader with your desired hyperparameters")
    print("4. Call prepare_data() to create train/val/test splits")
    print("5. Call build_model() to create the model")
    print("6. Call train() to train the model")
    print("7. Call backtest() to evaluate performance")
    print("\nSee example_usage.py for a complete example.")
