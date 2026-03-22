import os
import sys
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering.regime_detection.regime_detector import RegimeDetector, HMM_AVAILABLE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GenerateGlobalRegime")

def main():
    if not HMM_AVAILABLE:
        logger.error("hmmlearn not installed. Cannot generate regime states.")
        return

    merged_data_path = os.path.join(PROJECT_ROOT, "data", "processed", "merged", "all_merged_dataset.csv")
    if not os.path.exists(merged_data_path):
        logger.error(f"Merged dataset not found at {merged_data_path}")
        return

    logger.info("Loading merged dataset...")
    df = pd.read_csv(merged_data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Aggregate to create a global market index proxy
    logger.info("Aggregating data to form a global market proxy...")
    global_df = df.groupby('date').agg({'Close': 'mean', 'Volume': 'mean'}).reset_index()
    global_df = global_df.sort_values('date')
    
    # Needs to match what RegimeDetector expects: 'close' or 'Close'
    # It also expects 'Volume'
    
    detector = RegimeDetector(n_regimes=5)
    logger.info("Fitting HMM Regime Detector on global market proxy...")
    detector.fit(global_df)
    
    labels, idx = detector.predict(global_df)
    
    # Create the output dataframe
    regime_df = pd.DataFrame({
        'date': global_df['date'].iloc[idx],
        'regime': labels
    })
    
    out_dir = os.path.join(PROJECT_ROOT, "data", "features")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "regime_states.csv")
    
    regime_df.to_csv(out_path, index=False)
    logger.info(f"Saved global regime states to {out_path}")
    
    # Also save the model
    model_path = os.path.join(PROJECT_ROOT, "saved_models", "regime_model.pkl")
    detector.save(model_path)

if __name__ == "__main__":
    main()
