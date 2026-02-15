import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataset import SequenceDataset
from src.train import train_epoch, validate

# Import Transformer
try:
    from src.model import TransformerClassifier
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from transformer_model import TransformerClassifier


# ===== CONFIGURATION =====
DATA_DIR = Path("C:/Users/naina/OneDrive/Desktop/UIUC/Fall 2025/IE 421/IE 421 project/group_01_project/data")
OUTPUT_DIR = Path("C:/Users/naina/OneDrive/Desktop/UIUC/Fall 2025/IE 421/IE 421 project/group_01_project/outputs/transformer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load best config from Ray Tune or use defaults
TUNE_CONFIG_PATH = Path("C:/Users/naina/OneDrive/Desktop/UIUC/Fall 2025/IE 421/IE 421 project/group_01_project/outputs/ray_tune_transformer/best_config.json")

if TUNE_CONFIG_PATH.exists():
    print(f"Loading best hyperparameters from Ray Tune...")
    with open(TUNE_CONFIG_PATH, 'r') as f:
        best_config = json.load(f)
    
    SEQUENCE_LENGTH = best_config['sequence_length']
    BATCH_SIZE = best_config['batch_size']
    D_MODEL = best_config['d_model']
    NHEAD = best_config['nhead']
    NUM_LAYERS = best_config['num_layers']
    DIM_FEEDFORWARD = best_config['dim_feedforward']
    DROPOUT = best_config['dropout']
    LEARNING_RATE = best_config['learning_rate']
    
    print(f"  ✓ Loaded config from {TUNE_CONFIG_PATH}")
else:
    print(f"No tuned config found, using defaults...")
    SEQUENCE_LENGTH = 50
    BATCH_SIZE = 64
    D_MODEL = 128
    NHEAD = 8
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    LEARNING_RATE = 0.001

# Training config
MAX_EPOCHS = 100
PATIENCE = 15
GRADIENT_CLIP = 1.0

# Data paths
TRAIN_CSV = str(DATA_DIR / 'train_data.csv')
VAL_CSV = str(DATA_DIR / 'val_data.csv')
TEST_CSV = str(DATA_DIR / 'test_data.csv')


def train_model():
    """Main training function"""
    
    print("\n" + "="*70)
    print("TRANSFORMER MODEL TRAINING")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load features
    print("\nLoading features...")
    with open(DATA_DIR / 'selected_features.pkl', 'rb') as f:
        FEATURE_COLUMNS = pickle.load(f)
    print(f"  Number of features: {len(FEATURE_COLUMNS)}")
    
    # Print config
    print("\nHyperparameters:")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  d_model: {D_MODEL}")
    print(f"  nhead: {NHEAD}")
    print(f"  num_layers: {NUM_LAYERS}")
    print(f"  dim_feedforward: {DIM_FEEDFORWARD}")
    print(f"  dropout: {DROPOUT}")
    print(f"  learning_rate: {LEARNING_RATE}")
    
    # Validate config
    if D_MODEL % NHEAD != 0:
        raise ValueError(f"d_model ({D_MODEL}) must be divisible by nhead ({NHEAD})")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SequenceDataset(TRAIN_CSV, SEQUENCE_LENGTH, FEATURE_COLUMNS)
    val_dataset = SequenceDataset(VAL_CSV, SEQUENCE_LENGTH, FEATURE_COLUMNS)
    test_dataset = SequenceDataset(TEST_CSV, SEQUENCE_LENGTH, FEATURE_COLUMNS)
    
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\nInitializing model...")
    model = TransformerClassifier(
        input_size=len(FEATURE_COLUMNS),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{MAX_EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['accuracy'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            model_path = OUTPUT_DIR / 'best_transformer_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'hyperparameters': {
                    'sequence_length': SEQUENCE_LENGTH,
                    'batch_size': BATCH_SIZE,
                    'd_model': D_MODEL,
                    'nhead': NHEAD,
                    'num_layers': NUM_LAYERS,
                    'dim_feedforward': DIM_FEEDFORWARD,
                    'dropout': DROPOUT,
                    'learning_rate': LEARNING_RATE,
                    'input_size': len(FEATURE_COLUMNS)
                }
            }, model_path)
            
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s)")
            
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Test
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(OUTPUT_DIR / 'best_transformer_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'best_val_loss': float(best_val_loss),
        'best_epoch': int(checkpoint['epoch']),
        'hyperparameters': checkpoint['hyperparameters'],
        'training_history': {
            'train_losses': [float(x) for x in train_losses],
            'val_losses': [float(x) for x in val_losses],
            'val_accuracies': [float(x) for x in val_accuracies]
        }
    }
    
    with open(OUTPUT_DIR / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {OUTPUT_DIR / 'training_results.json'}")
    print(f"✓ Model saved to {OUTPUT_DIR / 'best_transformer_model.pth'}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70 + "\n")
    
    return model, results


if __name__ == '__main__':
    model, results = train_model()
