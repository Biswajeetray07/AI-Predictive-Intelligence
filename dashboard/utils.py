"""
Data loading utilities for the AI Predictive Intelligence Dashboard.
Reads directly from the project's data directories (no API server needed).
"""

import os
import glob
import json
import datetime
import pandas as pd
import numpy as np
import psutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_project_root():
    return PROJECT_ROOT


# ─── Overview KPIs ────────────────────────────────────────────────────────────

def count_total_records():
    """Count total rows across all raw CSVs."""
    total = 0
    for csv_file in glob.glob(os.path.join(PROJECT_ROOT, 'data', 'raw', '**', '*.csv'), recursive=True):
        try:
            # Fast row count without loading data
            with open(csv_file, 'r') as f:
                total += sum(1 for _ in f) - 1  # minus header
        except Exception:
            pass
    return total


def count_active_apis():
    """Count API keys configured in .env."""
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if not os.path.exists(env_path):
        return 0
    count = 0
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                if ('KEY' in key or 'TOKEN' in key or 'API' in key) and val.strip().strip('"').strip("'"):
                    count += 1
    return count


def count_saved_models():
    """Count model artifacts in saved_models/."""
    model_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    if not os.path.exists(model_dir):
        return 0
    return len([f for f in glob.glob(os.path.join(model_dir, '**', '*.pt'), recursive=True)])


def get_overview_kpis():
    """Return dict of overview KPIs."""
    return {
        'total_records': count_total_records(),
        'active_apis': count_active_apis(),
        'models_saved': count_saved_models(),
        'features_generated': _count_feature_files(),
        'data_sources': _count_data_sources(),
    }


def _count_feature_files():
    feat_dir = os.path.join(PROJECT_ROOT, 'data', 'features')
    if not os.path.exists(feat_dir):
        return 0
    return len(glob.glob(os.path.join(feat_dir, '*.parquet'))) + len(glob.glob(os.path.join(feat_dir, '*.csv')))


def _count_data_sources():
    raw_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
    if not os.path.exists(raw_dir):
        return 0
    sources = set()
    for d in os.listdir(raw_dir):
        if os.path.isdir(os.path.join(raw_dir, d)):
            sources.add(d)
    return len(sources)


# ─── Data Sources ─────────────────────────────────────────────────────────────

def get_data_sources_info():
    """Scan data/raw/ for all source directories and their stats."""
    raw_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
    sources = []
    if not os.path.exists(raw_dir):
        return sources

    for domain in sorted(os.listdir(raw_dir)):
        domain_path = os.path.join(raw_dir, domain)
        if not os.path.isdir(domain_path):
            continue

        files = glob.glob(os.path.join(domain_path, '**', '*.*'), recursive=True)
        data_files = [f for f in files if f.endswith(('.csv', '.parquet', '.json'))]
        total_size = sum(os.path.getsize(f) for f in data_files if os.path.exists(f))
        total_records = 0
        latest_modified = None

        for f in data_files:
            try:
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
                if latest_modified is None or mtime > latest_modified:
                    latest_modified = mtime
            except Exception:
                pass
            if f.endswith('.csv'):
                try:
                    with open(f, 'r') as fh:
                        total_records += sum(1 for _ in fh) - 1
                except Exception:
                    pass

        sources.append({
            'name': domain.replace('_', ' ').title(),
            'directory': domain,
            'files': len(data_files),
            'records': total_records,
            'size_mb': round(total_size / (1024 * 1024), 2),
            'last_updated': latest_modified.strftime('%Y-%m-%d %H:%M') if latest_modified else 'N/A',
            'status': 'Online' if data_files else 'No Data',
        })

    return sources


# ─── Datasets ─────────────────────────────────────────────────────────────────

def get_datasets_info():
    """Discover processed datasets in data/processed/."""
    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    datasets = []
    if not os.path.exists(processed_dir):
        return datasets

    for f in glob.glob(os.path.join(processed_dir, '**', '*.csv'), recursive=True):
        try:
            df = pd.read_csv(f, nrows=0)
            row_count_est = sum(1 for _ in open(f)) - 1
            datasets.append({
                'name': os.path.basename(f),
                'path': f,
                'records': row_count_est,
                'features': len(df.columns),
                'columns': list(df.columns),
                'size_mb': round(os.path.getsize(f) / (1024 * 1024), 2),
            })
        except Exception:
            pass

    # Also add parquet files
    for f in glob.glob(os.path.join(processed_dir, '**', '*.parquet'), recursive=True):
        try:
            df = pd.read_parquet(f)
            datasets.append({
                'name': os.path.basename(f),
                'path': f,
                'records': len(df),
                'features': len(df.columns),
                'columns': list(df.columns),
                'size_mb': round(os.path.getsize(f) / (1024 * 1024), 2),
            })
        except Exception:
            pass

    # Feature files
    feat_dir = os.path.join(PROJECT_ROOT, 'data', 'features')
    if os.path.exists(feat_dir):
        for f in glob.glob(os.path.join(feat_dir, '*.parquet')):
            try:
                df = pd.read_parquet(f)
                datasets.append({
                    'name': os.path.basename(f),
                    'path': f,
                    'records': len(df),
                    'features': len(df.columns),
                    'columns': list(df.columns),
                    'size_mb': round(os.path.getsize(f) / (1024 * 1024), 2),
                })
            except Exception:
                pass

    return datasets


# ─── Models ───────────────────────────────────────────────────────────────────

def get_model_info():
    """Discover saved model artifacts."""
    model_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    models = []
    if not os.path.exists(model_dir):
        return models

    for f in glob.glob(os.path.join(model_dir, '**', '*.pt'), recursive=True):
        name = os.path.basename(f).replace('.pt', '')
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        size_mb = round(os.path.getsize(f) / (1024 * 1024), 2)

        # Determine model type
        if 'nlp' in name.lower():
            model_type = 'NLP Multi-Task'
            algorithm = 'DeBERTa-v3 + Multi-Head'
        elif 'timeseries' in name.lower() or 'ts' in name.lower():
            model_type = 'Time Series'
            algorithm = 'Transformer'
        elif 'fusion' in name.lower():
            model_type = 'Fusion'
            algorithm = 'Cross-Attention Fusion'
        else:
            model_type = 'Unknown'
            algorithm = 'N/A'

        models.append({
            'name': name,
            'type': model_type,
            'algorithm': algorithm,
            'size_mb': size_mb,
            'trained_at': mtime.strftime('%Y-%m-%d %H:%M'),
            'path': f,
        })

    return models


# ─── System Monitoring ────────────────────────────────────────────────────────

def get_system_stats():
    """Get current system resource usage."""
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_used_gb': round(memory.used / (1024**3), 1),
        'memory_total_gb': round(memory.total / (1024**3), 1),
        'disk_percent': disk.percent,
        'disk_used_gb': round(disk.used / (1024**3), 1),
        'disk_total_gb': round(disk.total / (1024**3), 1),
    }


# ─── Pipeline Status ─────────────────────────────────────────────────────────

def get_pipeline_stages():
    """Infer pipeline status from file timestamps."""
    stages = []
    checks = [
        ('Data Collection', os.path.join(PROJECT_ROOT, 'data', 'raw')),
        ('Data Processing', os.path.join(PROJECT_ROOT, 'data', 'processed')),
        ('Feature Engineering', os.path.join(PROJECT_ROOT, 'data', 'features')),
        ('Model Training', os.path.join(PROJECT_ROOT, 'saved_models')),
        ('Merged Dataset', os.path.join(PROJECT_ROOT, 'data', 'processed', 'merged')),
    ]

    for name, path in checks:
        exists = os.path.exists(path)
        file_count = 0
        last_modified = None
        if exists:
            for f in glob.glob(os.path.join(path, '**', '*.*'), recursive=True):
                file_count += 1
                try:
                    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
                    if last_modified is None or mtime > last_modified:
                        last_modified = mtime
                except Exception:
                    pass

        if file_count > 0:
            status = 'Complete'
        elif exists:
            status = 'Empty'
        else:
            status = 'Not Started'

        stages.append({
            'name': name,
            'status': status,
            'files': file_count,
            'last_modified': last_modified.strftime('%Y-%m-%d %H:%M') if last_modified else 'N/A',
        })

    return stages


# ─── Stock/Financial Data ─────────────────────────────────────────────────────

def load_stock_data(ticker='AAPL'):
    """Load processed stock data for a given ticker."""
    path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'financial', 'stocks', f'{ticker}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'date': 'Date'})
        return df
    return None


def list_available_tickers():
    """List all available stock tickers."""
    stocks_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'financial', 'stocks')
    if not os.path.exists(stocks_dir):
        return []
    return sorted([f.replace('.csv', '') for f in os.listdir(stocks_dir) if f.endswith('.csv')])


# ─── Logs ─────────────────────────────────────────────────────────────────────

def get_recent_logs(n=50):
    """Read recent log lines from any .log files in the project."""
    log_files = glob.glob(os.path.join(PROJECT_ROOT, '**', '*.log'), recursive=True)
    lines = []
    for lf in log_files:
        try:
            with open(lf, 'r') as f:
                lines.extend(f.readlines()[-n:])
        except Exception:
            pass
    return lines[-n:] if lines else ['No log files found.']
