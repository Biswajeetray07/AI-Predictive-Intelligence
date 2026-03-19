"""
Real-time utilities for the AI Predictive Intelligence Dashboard.
Provides live API integration, caching, and metrics collection.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "predictive_intel_dev_key_2026"
PROMETHEUS_URL = "http://localhost:9090"

# ═══════════════════════════════════════════════════════════════════════════════
# API HEALTH & STATUS
# ═══════════════════════════════════════════════════════════════════════════════

def check_api_health() -> Tuple[bool, Dict]:
    """Check API server health."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/health",
            timeout=2,
            headers={"X-API-Key": API_KEY}
        )
        if resp.status_code == 200:
            return True, resp.json()
    except Exception as e:
        return False, {"error": str(e)}
    return False, {}

def check_service_health(url: str, service_name: str = "Service") -> Tuple[bool, str]:
    """Check generic service health."""
    try:
        resp = requests.get(url, timeout=2)
        if resp.status_code < 400:
            return True, f"{service_name} Online"
    except Exception as e:
        return False, f"{service_name} Offline: {str(e)}"
    return False, f"{service_name} Offline"

# ═══════════════════════════════════════════════════════════════════════════════
# PROMETHEUS METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def query_prometheus(query: str, timeout: int = 3) -> Optional[float]:
    """Query Prometheus metric."""
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=timeout
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data', {}).get('result'):
                return float(data['data']['result'][0]['value'][1])
    except Exception:
        pass
    return None

def get_metrics_summary() -> Dict:
    """Get comprehensive metrics summary."""
    return {
        'predictions_per_sec': query_prometheus("rate(predictions_total[5m])") or 0,
        'avg_latency_ms': (query_prometheus("avg(prediction_latency_seconds)") or 0) * 1000,
        'error_rate': (query_prometheus("rate(prediction_errors_total[5m])") or 0) * 100,
        'cache_hit_rate': (query_prometheus("rate(cache_hits_total[5m])") or 0) * 100,
        'active_models': query_prometheus("models_loaded") or 0,
    }

def get_metrics_timeseries(query: str, duration: str = "1h", step: str = "5m") -> Optional[pd.DataFrame]:
    """Get metrics time series from Prometheus."""
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={
                "query": query,
                "start": (datetime.now() - timedelta(hours=1)).timestamp(),
                "end": datetime.now().timestamp(),
                "step": step,
            },
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data', {}).get('result'):
                result = data['data']['result'][0]
                times = [datetime.fromtimestamp(int(t[0])) for t in result['values']]
                values = [float(t[1]) for t in result['values']]
                return pd.DataFrame({'timestamp': times, 'value': values})
    except Exception:
        pass
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION API CALLS
# ═══════════════════════════════════════════════════════════════════════════════

def get_prediction(ticker: str, date: str = None) -> Optional[Dict]:
    """Get single prediction."""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    try:
        resp = requests.post(
            f"{API_BASE_URL}/predict",
            json={"ticker": ticker, "date": date},
            timeout=10,
            headers={"X-API-Key": API_KEY}
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def get_batch_predictions(tickers: List[str], date: str = None) -> Optional[Dict]:
    """Get batch predictions for multiple tickers."""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            f"{API_BASE_URL}/predictions/batch",
            params={"tickers": ",".join(tickers), "dates": date},
            timeout=15,
            headers={"X-API-Key": API_KEY}
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def get_prediction_report() -> Optional[Dict]:
    """Get prediction accuracy report."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/predictions/report",
            timeout=10,
            headers={"X-API-Key": API_KEY}
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════

def get_models_info() -> Optional[Dict]:
    """Get all models information."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/info/models",
            timeout=5,
            headers={"X-API-Key": API_KEY}
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def get_data_sources_info() -> Optional[Dict]:
    """Get data sources information."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/info/data-sources",
            timeout=5,
            headers={"X-API-Key": API_KEY}
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_drift_report() -> Optional[Dict]:
    """Get data drift detection report."""
    try:
        # In a real scenario, this would come from an API endpoint
        # For now, we'll query it from logs or a database
        return {
            'feature_drift': {
                'AAPL_volume': {'psi': 0.08, 'status': 'NORMAL'},
                'AAPL_price': {'psi': 0.12, 'status': 'NORMAL'},
                'market_sentiment': {'psi': 0.35, 'status': 'ALERT'},
            },
            'prediction_drift': {
                'psi': 0.14,
                'status': 'NORMAL',
            },
            'last_update': datetime.now().isoformat(),
        }
    except Exception:
        pass
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION (For demo/testing)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_mock_predictions(tickers: List[str], days: int = 7) -> pd.DataFrame:
    """Generate mock prediction data for demo."""
    data = []
    base_price = 150
    for day in range(days):
        date = datetime.now() - timedelta(days=day)
        for ticker in tickers:
            price = base_price + np.random.randn() * 10
            data.append({
                'date': date,
                'ticker': ticker,
                'predicted_price': price,
                'confidence': np.random.uniform(0.7, 0.99),
                'signal': np.random.choice(['BUY', 'HOLD', 'SELL']),
            })
    return pd.DataFrame(data)

def generate_mock_metrics(hours: int = 24) -> pd.DataFrame:
    """Generate mock performance metrics for demo."""
    data = []
    for h in range(hours):
        timestamp = datetime.now() - timedelta(hours=h)
        data.append({
            'timestamp': timestamp,
            'predictions_per_sec': np.random.uniform(50, 200),
            'avg_latency_ms': np.random.uniform(30, 150),
            'error_rate': np.random.uniform(0, 0.5),
            'cpu_percent': np.random.uniform(20, 80),
            'memory_percent': np.random.uniform(30, 70),
        })
    return pd.DataFrame(data).sort_values('timestamp')

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def format_metric(value: float, unit: str = "") -> str:
    """Format metric value for display."""
    if value >= 1000000:
        return f"{value/1000000:.1f}M {unit}".strip()
    elif value >= 1000:
        return f"{value/1000:.1f}K {unit}".strip()
    else:
        return f"{value:.1f} {unit}".strip()

def get_status_color(status: str) -> str:
    """Get color for status indicator."""
    colors = {
        'ONLINE': '#1EA26F',
        'OFFLINE': '#FF4D6D',
        'WARNING': '#FFB800',
        'NORMAL': '#00F5D4',
        'ALERT': '#FF4D6D',
    }
    return colors.get(status, '#8B95A5')
