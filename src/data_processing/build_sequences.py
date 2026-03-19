import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def create_memmap_dataset(df: pd.DataFrame, split_name: str, features_cols: list, target_col: str, seq_length: int, out_dir: str):
    """
    Creates sequences from a dataframe, ticker by ticker, and saves them
    into memory-mapped numpy arrays to avoid MemoryError on large datasets.
    """
    print(f"Counting sequences for {split_name} split...")
    total_seqs = 0
    # Group by ticker to ensure sequences do not cross tickers
    tickers_data = {}
    for ticker, group in df.groupby('ticker'):
        group_len = len(group)
        if group_len > seq_length:
            total_seqs += (group_len - seq_length)
            tickers_data[ticker] = group
            
    if total_seqs == 0:
        print(f"Warning: No sequences generated for {split_name} split. Not enough data per ticker.")
        return

    num_features = len(features_cols)
    print(f"{split_name} split: pre-allocating memmap for {total_seqs} sequences of length {seq_length}...")
    
    # Pre-allocate memmap
    X_path = os.path.join(out_dir, f"X_{split_name}.npy")
    y_path = os.path.join(out_dir, f"y_{split_name}.npy")
    y_multi_path = os.path.join(out_dir, f"y_multi_{split_name}.npy")
    
    X_mmap = np.lib.format.open_memmap(X_path, mode='w+', dtype='float32', shape=(total_seqs, seq_length, num_features))
    y_mmap = np.lib.format.open_memmap(y_path, mode='w+', dtype='float32', shape=(total_seqs, 1))
    y_multi_mmap = np.lib.format.open_memmap(y_multi_path, mode='w+', dtype='float32', shape=(total_seqs, 3))
    
    metadata_records = []
    
    current_idx = 0
    print(f"Generating sliding windows for {split_name} split...")
    for ticker, group in tqdm(tickers_data.items(), desc=f"{split_name} generation"):
        # Extract features and targets as numpy arrays
        features_np = group[features_cols].values.astype(np.float32)
        
        # Target is the Close price (scaled or unscaled, depending on input)
        # We want to predict the next day's Close price
        target_np = group[target_col].values.astype(np.float32)

        # Vectorized sliding window creation using stride_tricks
        n_windows = len(features_np) - seq_length
        if n_windows <= 0:
            continue
            
        shape = (n_windows, seq_length, num_features)
        strides = (features_np.strides[0], features_np.strides[0], features_np.strides[1])
        X_windows = np.lib.stride_tricks.as_strided(features_np, shape=shape, strides=strides)
        
        # The target for a sequence x[t : t+seq_length] is the value at t+seq_length
        y_targets = target_np[seq_length:]
        
        # Multi-horizon targets (1d, 5d, 30d) safely bounded
        indices_5d = np.minimum(np.arange(seq_length, len(target_np)) + 4, len(target_np) - 1)
        indices_30d = np.minimum(np.arange(seq_length, len(target_np)) + 29, len(target_np) - 1)
        
        y_1d = target_np[seq_length:]
        y_5d = target_np[indices_5d]
        y_30d = target_np[indices_30d]
        y_targets_multi = np.stack([y_1d, y_5d, y_30d], axis=1)
        
        target_dates = group['date'].values[seq_length:]
        
        X_mmap[current_idx:current_idx + n_windows] = X_windows
        y_mmap[current_idx:current_idx + n_windows, 0] = y_targets
        y_multi_mmap[current_idx:current_idx + n_windows] = y_targets_multi
        
        # Track metadata
        metadata_records.extend([{
            'date': target_dates[i],
            'ticker': ticker
        } for i in range(n_windows)])
        
        current_idx += n_windows

    # Flush the changes to disk
    X_mmap.flush()
    y_mmap.flush()
    y_multi_mmap.flush()
    
    # Save metadata
    meta_df = pd.DataFrame(metadata_records)
    meta_path = os.path.join(out_dir, f"metadata_{split_name}.csv")
    meta_df.to_csv(meta_path, index=False)
    
    print(f"Saved {split_name} arrays to {out_dir}")
    print(f"X_{split_name} shape: {X_mmap.shape}, size: {X_mmap.nbytes / (1024**3):.2f} GB")
    print(f"y_{split_name} shape: {y_mmap.shape}, size: {y_mmap.nbytes / (1024**2):.2f} MB")
    print(f"y_multi_{split_name} shape: {y_multi_mmap.shape}, size: {y_multi_mmap.nbytes / (1024**2):.2f} MB")
    print(f"metadata_{split_name} saved to {meta_path} with {len(meta_df)} rows")

def main():
    merged_data_path = os.path.join(PROJECT_ROOT, "data", "processed", "merged", "all_merged_dataset.csv")
    out_dir = os.path.join(PROJECT_ROOT, "data", "processed", "model_inputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading merged dataset from {merged_data_path}...")
    
    # Try multiple strategies to handle CSV parsing issues
    df = None
    parsing_strategies = [
        {"on_bad_lines": "skip", "engine": "python"},  # Skip malformed lines
        {"on_bad_lines": "skip", "quoting": 3, "engine": "python"},  # With aggressive quoting
        {"engine": "python", "error_bad_lines": False},  # Ignore bad lines
    ]
    
    for strategy in parsing_strategies:
        try:
            print(f"  Trying parsing strategy: {strategy}")
            df = pd.read_csv(merged_data_path, **strategy)
            print(f"  ✅ Successfully loaded with strategy: {strategy}")
            break
        except Exception as e:
            print(f"  ⚠️ Strategy failed: {e}")
            continue
    
    if df is None:
        raise RuntimeError(f"Failed to load CSV with all strategies. Check {merged_data_path}")
    
    print(f"Loaded {len(df)} rows")
    print("Formatting and sorting dates...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values(by=['ticker', 'date'], inplace=True)
    
    # LEAKAGE FIX: Forward-fill only — no bfill to prevent future data leaking into past
    df = df.groupby('ticker').apply(lambda group: group.ffill()).reset_index(drop=True)
    df.fillna(0, inplace=True) # Any that are still completely NaN
    
    # Define features and target
    # CRITICAL: target_col must be EXCLUDED from features to prevent data leakage
    target_col = 'Close'
    # Also exclude columns that are near-perfect proxies of Close (data leakage)
    leaky_cols = ['BB_Upper', 'BB_Lower', 'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26']
    exclude_cols = ['date', 'ticker', target_col] + leaky_cols
    features_cols = [c for c in df.columns if c not in exclude_cols]
    
    # ── Feature Quality Gating ──
    # 1. Drop non-numeric columns (would crash StandardScaler)
    non_numeric = [c for c in features_cols if df[c].dtype == 'object']
    if non_numeric:
        print(f"  🗑️ Dropping {len(non_numeric)} non-numeric columns: {non_numeric[:5]}...")
        features_cols = [c for c in features_cols if c not in non_numeric]
    
    # 2. Drop columns that are >99% constant (zero-information)
    constant_cols = []
    for c in features_cols:
        if df[c].nunique() <= 1:
            constant_cols.append(c)
        elif df[c].std() < 1e-10:
            constant_cols.append(c)
    if constant_cols:
        print(f"  🗑️ Dropping {len(constant_cols)} constant/near-constant columns: {constant_cols[:5]}...")
        features_cols = [c for c in features_cols if c not in constant_cols]
    
    # 3. Drop columns that are >50% NaN
    high_nan_cols = [c for c in features_cols if df[c].isna().mean() > 0.5]
    if high_nan_cols:
        print(f"  🗑️ Dropping {len(high_nan_cols)} high-NaN columns (>50%): {high_nan_cols[:5]}...")
        features_cols = [c for c in features_cols if c not in high_nan_cols]
    
    print(f"\nTarget column: '{target_col}' (excluded from features)")
    print(f"Feature columns after quality gating: {len(features_cols)} features")
    print(f"  Sample features: {features_cols[:10]}...")
    
    print("Performing temporal split...")
    # Adjusting based on data range: Min: 2023-01-03, Max: 2026-03-11
    # Train: 2023-2024
    # Val: First half of 2025
    # Test: July 2025 onwards
    train_mask = df['date'] < '2025-01-01'
    val_mask = (df['date'] >= '2025-01-01') & (df['date'] < '2025-07-01')
    test_mask = df['date'] >= '2025-07-01'
    
    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()
    
    print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}")
    
    print("Fitting global StandardScaler on Training set features...")
    feature_scaler = StandardScaler()
    feature_scaler.fit(df_train[features_cols])
    
    target_scaler = StandardScaler()
    target_scaler.fit(df_train[[target_col]]) # 2D array needed for sklearn
    
    # Save scalers for inference later
    joblib.dump(feature_scaler, os.path.join(out_dir, "feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(out_dir, "target_scaler.pkl"))
    print("Saved scalers to", out_dir)
    
    print("Scaling Train, Val, and Test DataFrames...")
    # Transform features
    # CRITICAL: We also check for distribution mismatches between Train and Val/Test
    # to prevent gradient explosion.
    
    train_stats = df_train[features_cols].describe().T
    
    for split_name, split_df in [("Val", df_val), ("Test", df_test)]:
        if len(split_df) == 0:
            continue
            
        print(f"\nAnalyzing distribution for {split_name} split...")
        split_stats = split_df[features_cols].describe().T
        
        # Check for Mean and Std shifts
        mean_diff = (split_stats['mean'] - train_stats['mean']).abs()
        std_ratio = (split_stats['std'] / (train_stats['std'] + 1e-9))
        
        mismatched_means = mean_diff[mean_diff > 1.0].index.tolist()
        explosive_stds = std_ratio[std_ratio > 3.0].index.tolist() # 3x volatility increase
        
        if mismatched_means:
            print(f"  ⚠️ WARNING: High mean shift in {split_name} for features: {mismatched_means[:5]}... (and {len(mismatched_means)-5} more)")
        if explosive_stds:
            print(f"  🔥 WARNING: High volatility surge in {split_name} for features: {explosive_stds[:5]}... (and {len(explosive_stds)-5} more)")

    # Perform transformation
    df_train.loc[:, features_cols] = feature_scaler.transform(df_train[features_cols])
    df_val.loc[:, features_cols] = feature_scaler.transform(df_val[features_cols])
    df_test.loc[:, features_cols] = feature_scaler.transform(df_test[features_cols])
    
    # ── ROBUST CLIPPING ──
    # Clip Z-scores at +/- 10 standard deviations to prevent infinite loss
    print("Clipping feature Z-scores at [-10, 10] to ensure stability...")
    df_train.loc[:, features_cols] = df_train[features_cols].clip(-10, 10)
    df_val.loc[:, features_cols] = df_val[features_cols].clip(-10, 10)
    df_test.loc[:, features_cols] = df_test[features_cols].clip(-10, 10)
    
    # Scale the TARGET column
    df_train.loc[:, target_col] = target_scaler.transform(df_train[[target_col]])
    df_val.loc[:, target_col] = target_scaler.transform(df_val[[target_col]])
    df_test.loc[:, target_col] = target_scaler.transform(df_test[[target_col]])

    seq_length = 60
    
    # Create and save memmap numpy datasets
    create_memmap_dataset(df_train, "train", features_cols, target_col, seq_length, out_dir)
    create_memmap_dataset(df_val, "val", features_cols, target_col, seq_length, out_dir)
    create_memmap_dataset(df_test, "test", features_cols, target_col, seq_length, out_dir)
    
    print("\nSEQUENCE BUILDING COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
