import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import SequenceDataset
from src.model import LSTMClassifier
from src.train import train_epoch, validate

import ray
from ray import tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

# ===== CONFIGURATION =====
DATA_DIR = Path("~/Desktop/hft_gitlab/data").expanduser()
OUTPUT_DIR = Path("~/Desktop/hft_gitlab/outputs/ray_tune").expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EPOCHS = 100 # test with 5, normal is 100
NUM_SAMPLES = 20  # test with 2, normal is 20

# !!! To test using 5 epochs -> change grade_period to 2 !!!
# !!! Make sure to change it back for 100 epochs !!!

# ===== LOAD DATA =====
print("Loading data and features...")
with open(DATA_DIR / 'selected_features.pkl', 'rb') as f:
    FEATURE_COLUMNS = pickle.load(f)

print(f"Features: {len(FEATURE_COLUMNS)}")

# These will be accessed by all trials
TRAIN_CSV = str(DATA_DIR / 'train_data.csv')
VAL_CSV = str(DATA_DIR / 'val_data.csv')


def train_lstm_trial(config):
    """Training function called by Ray Tune for each trial"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets (reads CSV - fast)
    train_dataset = SequenceDataset(TRAIN_CSV, config['sequence_length'], FEATURE_COLUMNS)
    val_dataset = SequenceDataset(VAL_CSV, config['sequence_length'], FEATURE_COLUMNS)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = LSTMClassifier(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    for epoch in range(MAX_EPOCHS):
        
        # Train one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
    
            # Report metrics (ASHA uses these to decide early stopping)
        tune.report({
            'loss': val_metrics['loss'],
            'accuracy': val_metrics['accuracy'],
            'f1': val_metrics['f1'],
            'epoch': epoch
            })


def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Define hyperparameter search space
    config = {
        'sequence_length': tune.choice([30, 50, 100]),
        'batch_size': tune.choice([64, 128, 256]),
        'hidden_size': tune.choice([32, 64, 128]),
        'num_layers': tune.choice([1, 2]),
        'dropout': tune.choice([0.2, 0.3, 0.5]),
        'learning_rate': tune.loguniform(1e-4, 1e-2)
    }
    
    # ASHA Scheduler (early stopping)
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=MAX_EPOCHS,
        grace_period=10,      # Don't stop before epoch 10, test with 2
        reduction_factor=2    # Keep top 50% of trials each round
    )
    
    print("\n" + "="*60)
    print("STARTING RAY TUNE HYPERPARAMETER SEARCH")
    print("="*60)
    print(f"Testing {NUM_SAMPLES} configurations")
    print(f"Max epochs per trial: {MAX_EPOCHS}")
    print(f"Early stopping with ASHA scheduler")
    print("="*60 + "\n")
    
    # Run tuning
    tuner = tune.Tuner(
        tune.with_resources(
            train_lstm_trial,
            resources={"cpu": 4, "gpu": 1}  # 0.5 GPU = 2 trials per GPU, use if access to multiple gpu's
        ),                                  # default: cpu : 4, gpu = 1
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=NUM_SAMPLES,
        ),
        param_space=config,
    )
    
    results = tuner.fit()
    
    # Get best result
    best_result = results.get_best_result(metric="loss", mode="min")
    
    print("\n" + "="*60)
    print("RAY TUNE COMPLETE - BEST CONFIGURATION")
    print("="*60)
    print(f"\nBest config:")
    for key, value in best_result.config.items():
        print(f"  {key}: {value}")
    
    print(f"\nBest validation loss: {best_result.metrics['loss']:.4f}")
    print(f"Best validation accuracy: {best_result.metrics['accuracy']:.4f}")
    print(f"Best validation F1: {best_result.metrics['f1']:.4f}")
    
   # Save best config (no checkpoint since we didn't save them during training)
    import json
    with open(OUTPUT_DIR / 'best_config.json', 'w') as f:
        json.dump(best_result.config, f, indent=2)
    
    print(f"\nBest config saved to: {OUTPUT_DIR / 'best_config.json'}")
    print("\n" + "="*60)
    print("TO TRAIN FINAL MODEL:")
    print("="*60)
    print("Update run_training.py with these values:\n")
    for key, value in best_result.config.items():
        if isinstance(value, float):
            print(f"{key.upper()} = {value:.6f}")
        else:
            print(f"{key.upper()} = {value}")
        
    ray.shutdown()


if __name__ == '__main__':
    main()