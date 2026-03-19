import mlflow
import os
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
mlflow.set_tracking_uri(DB_PATH)

def analyze_runs():
    print(f"Analyzing MLflow database at: {DB_PATH}")
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    print(f"Found {len(experiments)} experiments.")
    
    for exp in experiments:
        print(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})")
        
        # Search for runs, ordered by start time desc
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=5
        )
        if not runs: # runs is a PagedList, check if empty
            print("  No runs found.")
            continue
            
        # runs is a PagedList of Run objects
        for run in runs:
            run_id = run.info.run_id
            run_name = run.info.run_name if run.info.run_name else 'Unnamed' # Use run.info.run_name, default to 'Unnamed'
            status = run.info.status
            # Convert start_time from milliseconds to a formatted string
            start_time = pd.to_datetime(run.info.start_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"\n  Run: {run_name} (ID: {run_id[:8]}...) | Status: {status} | Started: {start_time}")
            
            client = mlflow.tracking.MlflowClient()
            metrics = client.get_run(run_id).data.metrics
            
            # For HPO trials, we often want the history of the objective
            print("  Latest Metrics:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.6f}")
                
            # Specific deep dive for active training runs
            for metric_key in ['val_loss', 'train_loss']:
                if metric_key in metrics:
                    history = client.get_metric_history(run_id, metric_key)
                    if history:
                        print(f"    {metric_key} Full History:")
                        history_sorted = sorted(history, key=lambda x: x.step)
                        for m in history_sorted:
                            print(f"      Step {m.step}: {m.value:.6f}")

if __name__ == "__main__":
    analyze_runs()
