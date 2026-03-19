import os
import sys
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.pipelines.inference_pipeline import Predictor
from src.evaluation.metrics import (
    calculate_regression_metrics,
    calculate_financial_metrics,
    calculate_per_ticker_metrics,
    calculate_statistical_significance,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_backtest(subset_size=None):
    """
    Run the full inference pipeline over the test dataset to simulate performance.
    
    Enhanced with:
        - Per-ticker metric breakdown
        - Statistical significance testing (Diebold-Mariano)
        - Comprehensive JSON + CSV reporting
    
    Args:
        subset_size (int): Run on a smaller subset for quick validation if not None.
    """
    logging.info("Initializing Backtest Engine...")
    
    # 1. Load Predictor
    try:
        predictor = Predictor()
    except Exception as e:
        logging.error(f"Failed to load Predictor. Ensure models are trained. Error: {e}")
        return
        
    # 2. Load Test Data
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'model_inputs')
    x_test_path = os.path.join(data_dir, 'X_test.npy')
    y_test_path = os.path.join(data_dir, 'y_test.npy')
    meta_path = os.path.join(data_dir, 'metadata_test.csv')
    
    if not (os.path.exists(x_test_path) and os.path.exists(y_test_path)):
        logging.error("Test data not found. Please ensure data pipeline ran completely.")
        return
        
    X_test = np.load(x_test_path, mmap_mode='r')
    y_test = np.load(y_test_path, mmap_mode='r')
    
    metadata = pd.DataFrame()
    if os.path.exists(meta_path):
        metadata = pd.read_csv(meta_path)
    
    # Use subset if specified
    total_samples = len(X_test)
    if subset_size and subset_size < total_samples:
        X_test = X_test[:subset_size]
        y_test = y_test[:subset_size]
        if not metadata.empty:
            metadata = metadata.head(subset_size)
        total_samples = subset_size
        logging.info(f"Running quick validation backtest on {subset_size} samples.")
    else:
        logging.info(f"Running full backtest on {total_samples} samples.")

    # 3. Predict over timeline
    predictions = []
    actuals = []
    
    for i in range(total_samples):
        if (i+1) % 100 == 0:
            logging.info(f"Backtesting... {i+1}/{total_samples}")
            
        x_sample = X_test[i]
        y_true_scaled = y_test[i][0]
        
        y_true = y_true_scaled
        if 'target' in predictor.scalers:
            y_true = predictor.scalers['target'].inverse_transform([[y_true_scaled]])[0][0]
            
        results = predictor.predict(x_sample)
        y_pred = results['multi_horizon_predictions']['1d']
        
        predictions.append(y_pred)
        actuals.append(y_true)
        
    # 4. Calculate Aggregate Metrics
    logging.info("Calculating Performance Metrics...")
    reg_metrics = calculate_regression_metrics(actuals, predictions)
    fin_metrics = calculate_financial_metrics(actuals, predictions)
    
    # 5. Statistical Significance Testing
    stat_sig = {}
    try:
        stat_sig = calculate_statistical_significance(actuals, predictions)
        logging.info(f"Statistical significance: {stat_sig}")
    except Exception as e:
        logging.warning(f"Statistical significance test skipped: {e}")
    
    # 6. Per-Ticker Metrics
    per_ticker_results = {}
    if not metadata.empty and 'ticker' in metadata.columns:
        logging.info("Calculating per-ticker metrics...")
        try:
            per_ticker_results = calculate_per_ticker_metrics(
                actuals, predictions, metadata['ticker'].tolist()
            )
            
            # Log top and bottom performers
            ticker_summary = []
            for ticker, metrics in per_ticker_results.items():
                if 'MSE' in metrics:
                    ticker_summary.append({'ticker': ticker, 'mse': metrics['MSE'], 'n': metrics.get('n', 0)})
            
            if ticker_summary:
                ticker_summary.sort(key=lambda x: x['mse'])
                logging.info("\nTop 5 Best Tickers (lowest MSE):")
                for t in ticker_summary[:5]:
                    logging.info(f"  {t['ticker']}: MSE={t['mse']:.6f} (n={t['n']})")
                logging.info("Top 5 Worst Tickers (highest MSE):")
                for t in ticker_summary[-5:]:
                    logging.info(f"  {t['ticker']}: MSE={t['mse']:.6f} (n={t['n']})")
        except Exception as e:
            logging.warning(f"Per-ticker metrics failed: {e}")
    
    # 7. Compile Results
    out_results = {
        'timestamp': datetime.now().isoformat(),
        'samples': total_samples,
        'regression_metrics': reg_metrics,
        'financial_metrics': fin_metrics,
        'statistical_significance': stat_sig,
        'per_ticker_summary': {
            ticker: {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}
            for ticker, metrics in per_ticker_results.items()
        } if per_ticker_results else {},
    }
    
    logging.info("\n" + "="*50)
    logging.info("BACKTEST RESULTS")
    logging.info("="*50)
    logging.info("--- Regression Metrics ---")
    for k, v in reg_metrics.items():
        logging.info(f"  {k}: {v:.6f}")
    logging.info("--- Financial Metrics ---")
    for k, v in fin_metrics.items():
        logging.info(f"  {k}: {v:.6f}")
    if stat_sig:
        logging.info("--- Statistical Significance ---")
        for k, v in stat_sig.items():
            logging.info(f"  {k}: {v}")
    logging.info(f"--- Per-Ticker Coverage: {len(per_ticker_results)} tickers ---")
    logging.info("="*50)
    
    # 8. Save Results
    results_dir = os.path.join(PROJECT_ROOT, 'evaluation', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save JSON summary
    json_path = os.path.join(results_dir, 'backtest_summary.json')
    with open(json_path, 'w') as f:
        json.dump(out_results, f, indent=4, default=str)
        
    # Save CSV tracing (for charting later)
    df_out = pd.DataFrame({'Actual': actuals, 'Predicted': predictions})
    if not metadata.empty:
        df_out['Date'] = metadata['date'].values
        df_out['Ticker'] = metadata['ticker'].values
    
    csv_path = os.path.join(results_dir, 'backtest_trace.csv')
    df_out.to_csv(csv_path, index=False)
    
    # Save per-ticker report
    if per_ticker_results:
        ticker_df = pd.DataFrame(per_ticker_results).T
        ticker_df.index.name = 'ticker'
        ticker_report_path = os.path.join(results_dir, 'per_ticker_report.csv')
        ticker_df.to_csv(ticker_report_path)
        logging.info(f"Per-ticker report saved to {ticker_report_path}")
    
    logging.info(f"Saved formal results to {results_dir}")

if __name__ == "__main__":
    # If run cleanly, perform full backtest. For quick testing, pass an arg
    subset = 500 if len(sys.argv) > 1 and sys.argv[1] == '--quick' else None
    run_backtest(subset)

