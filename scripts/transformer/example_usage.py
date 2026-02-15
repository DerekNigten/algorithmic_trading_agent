"""
Example Usage: Transformer Trading Model with IEX DEEP Data
IE 421: Agentic AI in Financial Markets

This script demonstrates how to use the Transformer model with IEX DEEP data
from the UIUC Campus Cluster.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when running as a script
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.transformer_trading_model import TransformerTrader
import warnings
warnings.filterwarnings('ignore')


def load_iex_tick_data(filepath: str, symbol: str = 'AAPL') -> pd.DataFrame:
    """
    Load IEX DEEP tick data from the campus cluster format.
    
    Args:
        filepath: Path to tick data file (e.g., 'tick_AAPL_20240110.txt.gz')
        symbol: Stock symbol
        
    Returns:
        DataFrame with parsed IEX tick data
    """
    print(f"Loading IEX tick data from {filepath}...")
    
    # Load the data with the correct column names from the campus cluster format
    df = pd.read_csv(
        filepath,
        compression='gzip',
        header=None,
        names=['timestamp_capture', 'timestamp_event', 'seq_id', 'msg_type',
               'exchange', 'field1', 'price', 'size', 'field2', 'field3',
               'field4', 'field5'],
        on_bad_lines='skip'
    )
    
    # Convert timestamps to datetime
    df['timestamp_event'] = pd.to_datetime(df['timestamp_event'], unit='ns')
    df['timestamp_capture'] = pd.to_datetime(df['timestamp_capture'], unit='ns')
    
    # Sort by event timestamp
    df = df.sort_values('timestamp_event').reset_index(drop=True)
    
    print(f"Loaded {len(df):,} records")
    print(f"  - Trades (T): {len(df[df['msg_type'] == 'T']):,}")
    print(f"  - Price updates (P): {len(df[df['msg_type'] == 'P']):,}")
    
    return df


def aggregate_to_bars(df: pd.DataFrame, interval: str = '1min') -> pd.DataFrame:
    """
    Aggregate tick data into OHLCV bars.
    
    Args:
        df: DataFrame with tick data
        interval: Time interval for bars (e.g., '1min', '5min', '30S')
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Aggregating to {interval} bars...")
    
    # Filter for trades only (msg_type == 'T')
    trades = df[df['msg_type'] == 'T'].copy()
    
    if len(trades) == 0:
        raise ValueError("No trades found in data!")
    
    # Set timestamp as index for resampling
    trades.set_index('timestamp_event', inplace=True)
    
    # Create OHLCV bars
    ohlcv = pd.DataFrame()
    ohlcv['open'] = trades['price'].resample(interval).first()
    ohlcv['high'] = trades['price'].resample(interval).max()
    ohlcv['low'] = trades['price'].resample(interval).min()
    ohlcv['close'] = trades['price'].resample(interval).last()
    ohlcv['volume'] = trades['size'].resample(interval).sum()
    ohlcv['n_trades'] = trades['price'].resample(interval).count()
    
    # Forward fill missing values (for periods with no trades)
    ohlcv['close'] = ohlcv['close'].fillna(method='ffill')
    ohlcv['open'] = ohlcv['open'].fillna(ohlcv['close'])
    ohlcv['high'] = ohlcv['high'].fillna(ohlcv['close'])
    ohlcv['low'] = ohlcv['low'].fillna(ohlcv['close'])
    ohlcv['volume'] = ohlcv['volume'].fillna(0)
    ohlcv['n_trades'] = ohlcv['n_trades'].fillna(0)
    
    # Reset index to make timestamp a column
    ohlcv.reset_index(inplace=True)
    ohlcv.rename(columns={'timestamp_event': 'timestamp'}, inplace=True)
    
    # Drop rows with NaN (beginning of data)
    ohlcv = ohlcv.dropna()
    
    print(f"Created {len(ohlcv):,} bars")
    
    return ohlcv


def calculate_order_book_features(df: pd.DataFrame, interval: str = '1min') -> pd.DataFrame:
    """
    Calculate order book imbalance and spread features from price updates.
    
    Args:
        df: DataFrame with tick data
        interval: Time interval for aggregation
        
    Returns:
        DataFrame with order book features
    """
    print(f"Calculating order book features...")
    
    # Filter for price updates (msg_type == 'P')
    price_updates = df[df['msg_type'] == 'P'].copy()
    
    if len(price_updates) == 0:
        print("Warning: No price updates found in data")
        return pd.DataFrame()
    
    # Set timestamp as index
    price_updates.set_index('timestamp_event', inplace=True)
    
    # Calculate features (this is simplified - adjust based on actual field meanings)
    features = pd.DataFrame()
    features['avg_price'] = price_updates['price'].resample(interval).mean()
    features['price_volatility'] = price_updates['price'].resample(interval).std()
    features['n_price_updates'] = price_updates['price'].resample(interval).count()
    
    # Reset index
    features.reset_index(inplace=True)
    features.rename(columns={'timestamp_event': 'timestamp'}, inplace=True)
    
    return features


def prepare_ml_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for machine learning from OHLCV data.
    
    Args:
        ohlcv: DataFrame with OHLCV bars
        
    Returns:
        DataFrame with engineered features
    """
    print("Engineering features...")
    
    df = ohlcv.copy()
    
    # Use 'close' price for calculations
    df['price'] = df['close']
    
    # Returns
    df['return'] = df['price'].pct_change()
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    
    # Moving averages
    df['sma_5'] = df['price'].rolling(window=5).mean()
    df['sma_10'] = df['price'].rolling(window=10).mean()
    df['sma_20'] = df['price'].rolling(window=20).mean()
    df['sma_ratio'] = df['sma_5'] / df['sma_20']
    
    # Exponential moving averages
    df['ema_5'] = df['price'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['price'].ewm(span=10, adjust=False).mean()
    
    # Volume features
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-8)
    df['volume_change'] = df['volume'].pct_change()
    
    # Volatility
    df['volatility_5'] = df['return'].rolling(window=5).std()
    df['volatility_10'] = df['return'].rolling(window=10).std()
    df['volatility_20'] = df['return'].rolling(window=20).std()
    
    # High-Low range
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['hl_range_ma'] = df['hl_range'].rolling(window=5).mean()
    
    # RSI (Relative Strength Index)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['price'].ewm(span=12, adjust=False).mean()
    exp2 = df['price'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['price'].rolling(window=20).mean()
    bb_std = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Momentum
    df['momentum_5'] = df['price'] - df['price'].shift(5)
    df['momentum_10'] = df['price'] - df['price'].shift(10)
    
    # Rate of change
    df['roc_5'] = ((df['price'] - df['price'].shift(5)) / df['price'].shift(5)) * 100
    df['roc_10'] = ((df['price'] - df['price'].shift(10)) / df['price'].shift(10)) * 100
    
    # Trade intensity
    df['trade_intensity'] = df['n_trades'] / (df['n_trades'].rolling(window=20).mean() + 1e-8)
    
    # Target: Future return (what we want to predict)
    # Predict return 1 period ahead
    df['future_return'] = df['return'].shift(-1)
    
    # Drop NaN values
    df = df.dropna()
    
    print(f"Created {len(df.columns)} features")
    print(f"Dataset size after cleaning: {len(df):,} samples")
    
    return df


def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/training_history.png', dpi=300)
    print("✓ Training history plot saved!")


def plot_predictions(predictions, targets, n_samples=500):
    """Plot predictions vs actual values."""
    plt.figure(figsize=(14, 6))
    
    # Limit to n_samples for readability
    pred_plot = predictions[:n_samples]
    target_plot = targets[:n_samples]
    
    plt.plot(target_plot, label='Actual Returns', alpha=0.7, linewidth=1.5)
    plt.plot(pred_plot, label='Predicted Returns', alpha=0.7, linewidth=1.5)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Return', fontsize=12)
    plt.title('Predicted vs Actual Returns', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/predictions.png', dpi=300)
    print("✓ Predictions plot saved!")


def plot_cumulative_returns(signals, targets):
    """Plot cumulative returns from trading strategy."""
    # Calculate returns based on signals
    returns = signals[:-1] * targets[1:]
    cumulative_returns = np.cumsum(returns)
    
    # Also calculate buy-and-hold returns for comparison
    buy_hold_returns = np.cumsum(targets[1:])
    
    plt.figure(figsize=(14, 6))
    plt.plot(cumulative_returns, label='Transformer Strategy', linewidth=2)
    plt.plot(buy_hold_returns, label='Buy and Hold', linewidth=2, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.title('Strategy Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/cumulative_returns.png', dpi=300)
    print("✓ Cumulative returns plot saved!")


def plot_attention_weights(model, sample_sequence):
    """
    Visualize attention weights from the Transformer model.
    This helps understand which time steps the model focuses on.
    """
    import torch
    
    model.eval()
    with torch.no_grad():
        # Get attention weights from the first layer
        # Note: This is a simplified visualization
        pass  # TODO: Implement attention visualization


def main():
    """
    Main function demonstrating the complete workflow.
    """
    print("\n" + "="*70)
    print("TRANSFORMER TRADING MODEL - IEX DEEP DATA")
    print("="*70 + "\n")
    
    # ========================================================================
    # STEP 1: LOAD AND PREPARE DATA
    # ========================================================================
    print("STEP 1: Loading and preparing data...")
    print("-" * 70)
    
    # TODO: Replace with your actual file path
    # Example: '/path/to/tick_AAPL_20240110.txt.gz'
    tick_data_file = 'tick_AAPL_20240110.txt.gz'
    
    # Load tick data
    df_tick = load_iex_tick_data(tick_data_file, symbol='AAPL')
    
    # Aggregate to 1-minute bars (you can change this to 30S, 5min, etc.)
    df_ohlcv = aggregate_to_bars(df_tick, interval='1min')
    
    # Engineer features
    df_features = prepare_ml_features(df_ohlcv)
    
    print(f"\nData summary:")
    print(f"  - Date range: {df_features['timestamp'].min()} to {df_features['timestamp'].max()}")
    print(f"  - Total bars: {len(df_features):,}")
    print(f"  - Features: {len(df_features.columns)}")
    
    # ========================================================================
    # STEP 2: SELECT FEATURES FOR MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Selecting features for model...")
    print("-" * 70)
    
    # Define feature columns (exclude timestamp, price, and target)
    feature_columns = [
        # Returns
        'return', 'log_return',
        # Moving averages
        'sma_5', 'sma_10', 'sma_20', 'sma_ratio',
        'ema_5', 'ema_10',
        # Volume
        'volume', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'volume_change',
        # Volatility
        'volatility_5', 'volatility_10', 'volatility_20',
        # Price range
        'hl_range', 'hl_range_ma',
        # Technical indicators
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_position',
        # Momentum
        'momentum_5', 'momentum_10',
        'roc_5', 'roc_10',
        # Trade intensity
        'trade_intensity', 'n_trades'
    ]
    
    # Verify all features exist
    feature_columns = [col for col in feature_columns if col in df_features.columns]
    
    print(f"Selected {len(feature_columns)} features:")
    for i, feat in enumerate(feature_columns, 1):
        print(f"  {i:2d}. {feat}")
    
    # ========================================================================
    # STEP 3: INITIALIZE AND CONFIGURE MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Initializing Transformer model...")
    print("-" * 70)
    
    # Initialize the Transformer trader
    trader = TransformerTrader(
        sequence_length=50,      # Look back 50 time steps
        d_model=128,             # Model dimension
        n_heads=8,               # Number of attention heads
        n_layers=4,              # Number of Transformer layers
        d_ff=512,                # Feedforward dimension
        dropout=0.1,             # Dropout rate
        learning_rate=0.001,     # Learning rate
        batch_size=32            # Batch size
    )
    
    # ========================================================================
    # STEP 4: PREPARE DATA LOADERS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: Preparing data loaders...")
    print("-" * 70)
    
    train_loader, val_loader, test_loader = trader.prepare_data(
        df=df_features,
        feature_columns=feature_columns,
        target_column='future_return',
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # ========================================================================
    # STEP 5: BUILD MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: Building model...")
    print("-" * 70)
    
    trader.build_model(n_features=len(feature_columns))
    
    # ========================================================================
    # STEP 6: TRAIN MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: Training model...")
    print("-" * 70)
    
    train_losses, val_losses = trader.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=50,
        patience=10
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    # ========================================================================
    # STEP 7: BACKTEST ON TEST SET
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: Backtesting on test set...")
    print("-" * 70)
    
    results = trader.backtest(
        test_loader=test_loader,
        decision_threshold=0.0  # Adjust this threshold for buy/sell decisions
    )
    
    # Plot predictions and cumulative returns
    plot_predictions(results['predictions'], results['targets'])
    plot_cumulative_returns(results['signals'], results['targets'])
    
    # ========================================================================
    # STEP 8: SAVE MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 8: Saving model...")
    print("-" * 70)
    
    trader.save_model('/mnt/user-data/outputs/transformer_model.pth')
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n✓ Model trained and saved")
    print(f"✓ Backtest results:")
    print(f"    - Directional Accuracy: {results['directional_accuracy']:.2%}")
    print(f"    - Total Return: {results['total_return']:.4f}")
    print(f"    - Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"\n✓ Visualizations saved to /mnt/user-data/outputs/")
    print(f"    - training_history.png")
    print(f"    - predictions.png")
    print(f"    - cumulative_returns.png")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Run the main workflow
    main()
