import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def create_report():
    print("Generating Evaluation Report...")
    
    results_dir = os.path.join(PROJECT_ROOT, 'evaluation', 'results')
    summary_path = os.path.join(results_dir, 'backtest_summary.json')
    trace_path = os.path.join(results_dir, 'backtest_trace.csv')
    
    if not (os.path.exists(summary_path) and os.path.exists(trace_path)):
        print("Results not found. Please run evaluation/backtest.py first.")
        return
        
    # Load data
    with open(summary_path, 'r') as f:
        summary = json.load(f)
        
    trace_df = pd.read_csv(trace_path)
    
    # 1. Generate Visualization
    plt.figure(figsize=(12, 6))
    
    # If we have multiple tickers, let's just plot the first one for the summary
    if 'Ticker' in trace_df.columns:
        top_ticker = trace_df['Ticker'].value_counts().index[0]
        plot_df = trace_df[trace_df['Ticker'] == top_ticker].head(100) # Plot first 100 days
        plt.title(f"Model Predictions vs Actual (Sample: {top_ticker})")
    else:
        plot_df = trace_df.head(100)
        plt.title("Model Predictions vs Actual (First 100 Samples)")
        
    plt.plot(plot_df['Actual'].values, label='Actual Change', color='black', alpha=0.7)
    plt.plot(plot_df['Predicted'].values, label='AI Predicted Change', color='blue', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    chart_path = os.path.join(results_dir, 'prediction_vs_actual.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved chart to {chart_path}")
    
    # 2. Generate Markdown Report
    report_path = os.path.join(results_dir, 'evaluation_report.md')
    
    reg_met = summary.get('regression_metrics', {})
    fin_met = summary.get('financial_metrics', {})
    
    md_content = f"""# AI-Predictive-Intelligence: Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This report details the performance of the Deep Fusion model over the historical test dataset ({summary.get('samples')} samples).

## Performance Metrics

### Financial Trading Metrics
| Metric | Score | Description |
| :--- | :--- | :--- |
| **Directional Accuracy** | `{fin_met.get('Directional_Accuracy', 0) * 100:.2f}%` | Percentage of times the AI correctly guessed the market direction (Up vs Down). |
| **Simulated Sharpe Ratio** | `{fin_met.get('Simulated_Sharpe_Ratio', 0):.2f}` | Risk-adjusted return of a simulated strategy following the AI's predictions. |
| **Mean Strategy Return** | `{fin_met.get('Mean_Strategy_Return', 0):.6f}` | Average return per trade. |

### Regression Error Metrics
| Metric | Value |
| :--- | :--- |
| **RMSE** (Root Mean Squared Error) | `{reg_met.get('RMSE', 0):.6f}` |
| **MAE** (Mean Absolute Error)     | `{reg_met.get('MAE', 0):.6f}` |
| **MSE** (Mean Squared Error)      | `{reg_met.get('MSE', 0):.6f}` |

## Visualization
![Prediction vs Actual Comparison](prediction_vs_actual.png)

*Note: The chart above shows a 100-day sample comparison between the AI's predicted price movement and the actual market movement.*
"""

    with open(report_path, 'w') as f:
        f.write(md_content)
        
    print(f"Saved Markdown report to {report_path}")

if __name__ == "__main__":
    create_report()
