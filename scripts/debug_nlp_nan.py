import torch
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.nlp.model import MultiTaskNLPModel
from src.models.nlp.tokenizer import NLPTokenizer
from src.training.nlp.train import NLPDataset

def debug_nan():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = NLPTokenizer()
    model = MultiTaskNLPModel(model_name=tokenizer.model_name)
    model = model.to(device)
    model.eval()

    # Create dummy data
    dummy_text = ["This is a test sentence for financial sentiment.", "Market is crashing!"]
    df = pd.DataFrame({
        'text': dummy_text,
        'source_id': [0, 1],
        'day_of_week': [0, 1],
        'month': [1, 1],
        'date': ['2026-03-19', '2026-03-19']
    })

    dataset = NLPDataset(df, tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    batch = next(iter(loader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    source_ids = batch['source_ids'].to(device)
    day_of_week = batch['day_of_week'].to(device)
    month = batch['month'].to(device)

    print("Checking inputs for NaNs...")
    print(f"input_ids: {torch.isnan(input_ids.float()).any()}")
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, source_ids, day_of_week, month)
    
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"Output '{k}' has NaNs: {torch.isnan(v).any()}")
            if torch.isnan(v).any():
                print(f"First 5 values of '{k}': {v[:5]}")

if __name__ == "__main__":
    debug_nan()
