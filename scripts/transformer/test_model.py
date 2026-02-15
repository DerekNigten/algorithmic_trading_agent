"""
Test Script: Verify Transformer Model Works
IE 421: Agentic AI in Financial Markets

This script tests the Transformer model with synthetic data to ensure
everything is working correctly before using real IEX data.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when running as a script
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.transformer_trading_model import TransformerTrader
import torch


def generate_synthetic_price_data(n_samples=5000, seed=42):
    """
    Generate synthetic price data that mimics real trading patterns.
    
    Args:
        n_samples: Number of data points to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    np.random.seed(seed)
    
    # Generate timestamps
    timestamps = pd.date_range('2024-01-01 09:30:00', periods=n_samples, freq='1min')
    
    # Generate price with trend and noise
    trend = np.linspace(100, 110, n_samples)
    noise = np.random.normal(0, 0.5, n_samples)
    sine_wave = 2 * np.sin(np.linspace(0, 10*np.pi, n_samples))
    
    close_price = trend + noise + sine_wave
    
    # Generate OHLCV
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': close_price + np.random.uniform(-0.2, 0.2, n_samples),
        'high': close_price + np.random.uniform(0, 0.5, n_samples),
        'low': close_price - np.random.uniform(0, 0.5, n_samples),
        'close': close_price,
        'volume': np.random.randint(1000, 10000, n_samples),
        'n_trades': np.random.randint(10, 100, n_samples)
    })
    
    return df


def engineer_simple_features(df):
    """
    Create a simplified set of features for testing.
    """
    df = df.copy()
    
    # Price features
    df['return'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # Volume features
    df['volume_ma'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    
    # Volatility
    df['volatility'] = df['return'].rolling(window=10).std()
    
    # Target
    df['future_return'] = df['return'].shift(-1)
    
    # Drop NaN
    df = df.dropna()
    
    return df


def run_test():
    """
    Run a complete test of the Transformer model.
    """
    print("\n" + "="*70)
    print("TRANSFORMER MODEL TEST")
    print("="*70 + "\n")
    
    # Check PyTorch and device
    print("1. Checking PyTorch installation...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    # Generate synthetic data
    print("\n2. Generating synthetic data...")
    df_ohlcv = generate_synthetic_price_data(n_samples=5000)
    print(f"   Generated {len(df_ohlcv):,} data points")
    
    # Engineer features
    print("\n3. Engineering features...")
    df_features = engineer_simple_features(df_ohlcv)
    print(f"   Created dataset with {len(df_features):,} samples")
    
    # Define features
    feature_columns = ['return', 'sma_5', 'sma_20', 'volume_ratio', 'volatility']
    print(f"   Using {len(feature_columns)} features: {feature_columns}")
    
    # Initialize model
    print("\n4. Initializing Transformer model...")
    trader = TransformerTrader(
        sequence_length=30,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
        learning_rate=0.001,
        batch_size=32
    )
    
    # Prepare data
    print("\n5. Preparing data loaders...")
    train_loader, val_loader, test_loader = trader.prepare_data(
        df=df_features,
        feature_columns=feature_columns,
        target_column='future_return',
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # Build model
    print("\n6. Building model...")
    trader.build_model(n_features=len(feature_columns))
    
    # Train model (just a few epochs for testing)
    print("\n7. Training model (5 epochs)...")
    train_losses, val_losses = trader.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=5,
        patience=3
    )
    
    print(f"\n   Final training loss: {train_losses[-1]:.6f}")
    print(f"   Final validation loss: {val_losses[-1]:.6f}")
    
    # Backtest
    print("\n8. Running backtest...")
    results = trader.backtest(test_loader, decision_threshold=0.0)
    
    # Save model
    print("\n9. Saving model...")
    trader.save_model('/home/claude/test_model.pth')
    
    # Test loading
    print("\n10. Testing model loading...")
    trader2 = TransformerTrader()
    trader2.load_model('/home/claude/test_model.pth', n_features=len(feature_columns))
    print("    Model loaded successfully!")
    
    # Summary
    print("\n" + "="*70)
    print("TEST COMPLETE! ✓")
    print("="*70)
    print("\nAll components are working correctly:")
    print("  ✓ Data generation")
    print("  ✓ Feature engineering")
    print("  ✓ Model initialization")
    print("  ✓ Training pipeline")
    print("  ✓ Backtesting")
    print("  ✓ Model saving/loading")
    print("\nYou're ready to use real IEX data!")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    results = run_test()
