import sys
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader
import pickle

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataset import SequenceDataset
from src.model import TransformerClassifier

# ===== CONFIGURATION =====
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "outputs" / "transformer"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_predictions():
    """Generate predictions matching team format"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load features
    print("\nLoading features...")
    with open(DATA_DIR / 'selected_features.pkl', 'rb') as f:
        FEATURE_COLUMNS = pickle.load(f)
    print(f"  Number of features: {len(FEATURE_COLUMNS)}")
    
    # Load trained model
    print("\nLoading trained model...")
    checkpoint = torch.load(MODEL_DIR / 'best_transformer_model.pth', map_location=device)
    hyperparams = checkpoint['hyperparameters']
    
    model = TransformerClassifier(
        input_size=hyperparams['input_size'],
        d_model=hyperparams['d_model'],
        nhead=hyperparams['nhead'],
        num_layers=hyperparams['num_layers'],
        dim_feedforward=hyperparams['dim_feedforward'],
        dropout=hyperparams['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded")
    
    # Process each dataset
    datasets = {
        'train': str(DATA_DIR / 'train_data.csv'),
        'val': str(DATA_DIR / 'val_data.csv'),
        'test': str(DATA_DIR / 'test_data.csv')
    }
    
    all_actuals = []
    all_predicted = []
    all_prob_down = []
    all_prob_neutral = []
    all_prob_up = []
    
    for split_name, csv_path in datasets.items():
        print(f"\nGenerating predictions for {split_name}...")
        
        # Create dataset
        dataset = SequenceDataset(
            csv_path, 
            hyperparams['sequence_length'], 
            FEATURE_COLUMNS
        )
        
        print(f"  Total samples: {len(dataset):,}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=256,  # Larger batch size for faster inference
            shuffle=False,
            num_workers=0
        )
        
        print(f"  Total batches: {len(dataloader):,}")
        
        # Generate predictions
        batch_num = 0
        with torch.no_grad():
            for x, y in dataloader:
                batch_num += 1
                
                # Print progress every 100 batches
                if batch_num % 100 == 0:
                    print(f"    Processed {batch_num}/{len(dataloader)} batches ({batch_num/len(dataloader)*100:.1f}%)")
                
                x = x.to(device)
                output = model(x)
                
                # Get probabilities
                probs = torch.softmax(output, dim=1)
                
                # Get predicted class
                pred_class = torch.argmax(output, dim=1)
                
                # Store results
                all_actuals.extend(y.cpu().numpy())
                all_predicted.extend(pred_class.cpu().numpy())
                all_prob_down.extend(probs[:, 0].cpu().numpy())
                all_prob_neutral.extend(probs[:, 1].cpu().numpy())
                all_prob_up.extend(probs[:, 2].cpu().numpy())
        
        print(f"  ✓ Completed {split_name}! Generated {len(dataset):,} predictions")
    
    # Create DataFrame matching team format
    print("\nCreating final prediction file...")
    df = pd.DataFrame({
        'actual': all_actuals,
        'predicted': all_predicted,
        'prob_down': all_prob_down,
        'prob_neutral': all_prob_neutral,
        'prob_up': all_prob_up
    })
    
    # Save to CSV
    output_path = OUTPUT_DIR / 'transformer_predictions.csv'
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ PREDICTIONS SAVED!")
    print(f"{'='*70}")
    print(f"Location: {output_path}")
    print(f"Total predictions: {len(df):,}")
    print(f"\nSample output:")
    print(df.head(10))
    print(f"\nPrediction distribution:")
    print(df['predicted'].value_counts().sort_index())
    
    return output_path


if __name__ == '__main__':
    generate_predictions()