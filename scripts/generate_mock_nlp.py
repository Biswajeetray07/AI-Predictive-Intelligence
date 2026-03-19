import numpy as np
import pandas as pd
import os
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'data', 'features')
os.makedirs(FEATURES_DIR, exist_ok=True)

def main():
    # Try to find the actual number of samples from X_val or X_train to match alignment
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'model_inputs')
    total_samples = 3095 # Fallback default
    
    if os.path.exists(os.path.join(data_dir, 'y_val.npy')):
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        total_samples = len(y_val)
        print(f"Detected {total_samples} samples from y_val.npy")
    else:
        # Check if we have merged dataset to guess length
        merged_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'merged', 'all_merged_dataset.csv')
        if os.path.exists(merged_path):
            # Just a rough count for mock purposes
            total_samples = 31096 # Matching the user's latest log: (31096, 64)
            print(f"Using log-matched sample count: {total_samples}")

    print(f"Generating mock NLP data for {total_samples} samples...")

    # Mock NLP Embeddings (768-D)
    nlp_embeddings = np.random.randn(total_samples, 768).astype(np.float32)
    np.save(os.path.join(FEATURES_DIR, 'nlp_embeddings.npy'), nlp_embeddings)

    # Mock TS Embeddings (64-D) - match the detected dim from users log (31096, 64)
    # We only generate this if it doesn't exist, as real TS training might have already saved it
    ts_emb_path = os.path.join(FEATURES_DIR, 'ts_embeddings.npy')
    if not os.path.exists(ts_emb_path):
        ts_embeddings = np.random.randn(total_samples, 64).astype(np.float32)
        np.save(ts_emb_path, ts_embeddings)
        print(f"Generated mock TS embeddings since {ts_emb_path} was missing.")

    # Mock NLP Signals (daily aggregated)
    dates = pd.date_range(start='2023-01-01', periods=total_samples, freq='D')
    tickers = ['AAPL'] * total_samples # Mock ticker

    signals_df = pd.DataFrame({
        'date': dates,
        'ticker': tickers,
        'sentiment_positive': np.random.rand(total_samples),
        'sentiment_negative': np.random.rand(total_samples),
        'sentiment_neutral': np.random.rand(total_samples),
        'event_policy': np.random.rand(total_samples),
    })
    signals_df.to_parquet(os.path.join(FEATURES_DIR, 'nlp_signals.parquet'))

    # Save date index for alignment
    pd.DataFrame({'date': dates, 'ticker': tickers}).to_csv(
        os.path.join(FEATURES_DIR, 'nlp_embedding_dates.csv'), index=False
    )

    print(f"Mock features generated successfully in {FEATURES_DIR}")

if __name__ == "__main__":
    main()
