import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import SequenceDataset
from src.model import LSTMClassifier
from src.model import TCNClassifier
from src.train import train_epoch, validate

MODEL_TYPE = 'TCN' #LSTM or TCN

# ===== HYPERPARAMETERS (EDIT THESE) =====
SEQUENCE_LENGTH = 50
BATCH_SIZE = 128
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 100 # test with 2, 100 is normal

# Paths
DATA_DIR = Path("~/Desktop/hft_gitlab/data").expanduser()
OUTPUT_DIR = Path("~/Desktop/hft_gitlab/outputs").expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load feature names
    with open(DATA_DIR / 'selected_features.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    
    print(f"Loaded {len(feature_columns)} features")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = SequenceDataset(
        DATA_DIR / 'train_data.csv',
        SEQUENCE_LENGTH,
        feature_columns
    )
    val_dataset = SequenceDataset(
        DATA_DIR / 'val_data.csv',
        SEQUENCE_LENGTH,
        feature_columns
    )
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Important for time series!
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    if MODEL_TYPE == 'LSTM':
        model = LSTMClassifier(
            input_size=len(feature_columns),
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
    ).to(device)
    elif MODEL_TYPE == 'TCN':
        model = TCNClassifier(
            input_size=len(feature_columns),
            num_channels=[HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE],  # 3 layers
            kernel_size=3,
            dropout=DROPOUT
        ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    best_val_loss = float('inf')
    results = []
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")
        
        # Save results
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics
        })
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pt')
            print("✓ Saved best model")
    
    # Save training history
    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("  - best_model.pt")
    print("  - training_history.json")

if __name__ == '__main__':
    main()