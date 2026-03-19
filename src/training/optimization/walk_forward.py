import numpy as np
import torch
import logging

class WalkForwardSplitter:
    """
    Splits time-series data into sequential chunks to prevent data leakage during CV.
    
    Example (n_splits=3)
    Fold 1: Train [0% - 40%], Val [40% - 60%]
    Fold 2: Train [0% - 60%], Val [60% - 80%]
    Fold 3: Train [0% - 80%], Val [80% - 100%]
    """
    def __init__(self, n_splits=3, min_train_size_ratio=0.4):
        self.n_splits = n_splits
        self.min_train_size_ratio = min_train_size_ratio
        
    def split(self, total_samples):
        splits = []
        min_train_samples = int(total_samples * self.min_train_size_ratio)
        val_size = (total_samples - min_train_samples) // self.n_splits
        
        for i in range(self.n_splits):
            train_start = 0
            train_end = min_train_samples + i * val_size
            val_start = train_end
            val_end = val_start + val_size
            
            # Ensure the last fold takes any remainder
            if i == self.n_splits - 1:
                val_end = total_samples
                
            splits.append((range(train_start, train_end), range(val_start, val_end)))
            
        return splits

def get_subset_dataloaders(full_dataset, train_indices, val_indices, batch_size=64):
    """Create subset DataLoaders given indices."""
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
