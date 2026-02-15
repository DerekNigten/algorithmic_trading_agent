import torch
from torch.utils.data import Dataset
import pandas as pd

class SequenceDataset(Dataset):
    """Creates sequences from CSV for LSTM training"""
    
    def __init__(self, csv_path, sequence_length, feature_columns):
        """
        Args:
            csv_path: Path to train/val/test CSV
            sequence_length: Number of past events to use (e.g., 50)
            feature_columns: List of feature names
        """
        self.df = pd.read_csv(csv_path)
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        
        # Extract features and labels
        self.features = self.df[feature_columns].values
        self.labels = self.df['label'].values
        
    def __len__(self):
        # Can't create sequences for first sequence_length rows
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of past events
        x = self.features[idx:idx + self.sequence_length]
        
        # Get label at current time
        y = self.labels[idx + self.sequence_length]
        
        return torch.FloatTensor(x), torch.LongTensor([y]).squeeze()