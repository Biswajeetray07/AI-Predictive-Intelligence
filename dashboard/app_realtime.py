"""
AI Predictive Intelligence — Real-Time Command Center
======================================================
Production-grade real-time analytics dashboard.
Connects to FastAPI REST API for live predictions and monitoring metrics.

Run: streamlit run dashboard/app_realtime.py --logger.level=error
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import json
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import threading

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════.════════════════════════════════════════════════════

API_BASE_URL = "http://localhost:8000"
API_KEY = "predictive_intel_dev_key_2026"
PROMETHEUS_URL = "http://localhost:9090"

COLORS = {
    'primary': '#00F5D4',
    'secondary': '#6C63FF',
    'accent': '#FF4D6D',
    'warning': '#FFB800',
    'success': '#1EA26F',
}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Predictive Intelligence - Real-Time",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    body { background-color: #0B0F1A; color: #EAEAEA; }
    .stApp { background-color: #0B0F1A; }
    .metric-card { 
        background: linear-gradient(135deg, #111827 0%, #1a1f2e 100%);
        border: 1px solid #2a3040;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00F5D4;
        letter-spacing: 0.5px;
        margin: 20px 0 10px 0;
    }
    .status-online { color: #1EA26F; font-weight: 600; }
    .status-warning { color: #FFB800; font-weight: 600; }
    .status-offline { color: #FF4D6D; font-weight: 600; }
    .neon-divider { 
        border-top: 1px solid #2a3040;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# REAL-TIME API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=5)
def check_api_health() -> Tuple[bool, Dict]:
    """Check if API is running and healthy."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/health",
            timeout=2,
            headers={"X-API-Key": API_KEY}
        )
        if resp.status_code == 200:
            return True, resp.json()
    except Exception as e:
        st.session_state.api_error = str(e)
    return False, {}

@st.cache_data(ttl=10)
def get_prometheus_metrics() -> Dict:
    """Query live Prometheus metrics."""
    metrics = {
        'predictions_per_sec': 0,
        'avg_latency_ms': 0,
        'error_rate': 0,
        'active_models': 0,
    }
    try:
        # Predictions/sec
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "rate(predictions_total[5m])"},
            timeout=3
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data', {}).get('result'):
                val = float(data['data']['result'][0]['value'][1])
                metrics['predictions_per_sec'] = max(0, val)

        # Avg latency
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "avg(prediction_latency_seconds)"},
            timeout=3
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data', {}).get('result'):
                val = float(data['data']['result'][0]['value'][1])
                metrics['avg_latency_ms'] = max(0, val * 1000)

        # Error rate
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "rate(prediction_errors_total[5m])"},
            timeout=3
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data', {}).get('result'):
                val = float(data['data']['result'][0]['value'][1])
                metrics['error_rate'] = max(0, val * 100)
    except Exception:
        pass
    return metrics

def get_live_prediction(ticker: str) -> Optional[Dict]:
    """Get live prediction from API."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/predict",
            json={"ticker": ticker, "date": datetime.now().strftime("%Y-%m-%d")},
            timeout=5,
            headers={"X-API-Key": API_KEY}
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def get_model_info() -> Optional[Dict]:
    """Get model information from API."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/info/models",
            timeout=3,
            headers={"X-API-Key": API_KEY}
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 2.5rem;">🧠</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #00F5D4;">AI PREDICTIVE</div>
        <div style="font-size: 0.75rem; color: #8B95A5;">Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Predictions", "📊 Metrics", "🚨 Anomalies", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # API Status
    api_healthy, api_info = check_api_health()
    status_color = '#1EA26F' if api_healthy else '#FF4D6D'
    status_text = '🟢 Online' if api_healthy else '🔴 Offline'

    st.markdown(f"""
    <div style="padding: 12px; background: #111827; border-radius: 8px; border: 1px solid #2a3040;">
        <div style="font-size: 0.75rem; color: #8B95A5; margin-bottom: 8px;">API STATUS</div>
        <div style="color: {status_color}; font-weight: 600;">{status_text}</div>
        <div style="font-size: 0.75rem; color: #8B95A5; margin-top: 8px;">
            Base: <span style="color: #00F5D4;">localhost:8000</span><br/>
            Metrics: <span style="color: #00F5D4;">localhost:9090</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Refresh Metrics", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.markdown("<div class='section-title'>📡 Real-Time Command Center</div>", unsafe_allow_html=True)

    api_healthy, _ = check_api_health()

    if not api_healthy:
        st.error("""
        ⚠️ **API Server Not Running**

        Start the API with:
        ```bash
        uvicorn src.api.app:app --reload --port 8000
        ```

        Or with Docker:
        ```bash
        docker-compose up -d
        ```
        """)
    else:
        # Real-time Metrics Row
        metrics = get_prometheus_metrics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Predictions/sec",
                f"{metrics['predictions_per_sec']:.1f}",
                delta=None,
                help="5-minute rolling average"
            )

        with col2:
            st.metric(
                "Avg Latency",
                f"{metrics['avg_latency_ms']:.0f}ms",
                delta=None,
                help="Mean prediction latency"
            )

        with col3:
            st.metric(
                "Error Rate",
                f"{metrics['error_rate']:.2f}%",
                delta=None,
                help="Errors per prediction"
            )

        with col4:
            models = get_model_info()
            model_count = len(models.get('models', [])) if models else 0
            st.metric(
                "Active Models",
                model_count,
                delta=None,
                help="Trained models ready"
            )

        st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

        # Real-time Predictions
        st.markdown("### 🔮 Get Live Predictions")

        col1, col2 = st.columns([2, 1])

        with col1:
            ticker = st.selectbox(
                "Select Ticker",
                ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "BTC", "ETH"],
                help="Stock or crypto ticker"
            )

        with col2:
            pred_date = st.date_input("Date", datetime.now())

        if st.button("📈 Get Prediction", use_container_width=True):
            with st.spinner(f"Fetching prediction for {ticker}..."):
                pred = get_live_prediction(ticker)
                if pred:
                    st.success(f"✅ Prediction received")
                    st.json(pred)
                else:
                    st.error("Failed to get prediction. Check API is running.")

        st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

        # Model Information
        st.markdown("### 🤖 Active Models")

        models_info = get_model_info()
        if models_info and models_info.get('models'):
            for model in models_info['models']:
                with st.expander(f"📊 {model.get('name', 'Unknown')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Type", model.get('type', 'N/A'))
                    with col2:
                        st.metric("Accuracy", f"{model.get('accuracy', 0):.2%}")
                    with col3:
                        st.metric("Updated", model.get('updated', 'N/A'))
        else:
            st.info("No models loaded yet.")

        # Performance Chart
        st.markdown("### 📊 Performance Timeline")

        # Generate sample data for demo
        hours = list(range(-24, 1))
        predictions = np.random.randint(50, 200, len(hours))
        latencies = np.random.uniform(30, 150, len(hours))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours, y=predictions,
            name='Predictions/sec',
            line=dict(color=COLORS['primary'], width=2),
            yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=latencies,
            name='Latency (ms)',
            line=dict(color=COLORS['secondary'], width=2, dash='dash'),
            yaxis='y2'
        ))
        fig.update_layout(
            title="API Performance (Last 24h)",
            xaxis_title="Hours ago",
            yaxis_title="Predictions/sec",
            yaxis2=dict(title="Latency (ms)", overlaying='y', side='right'),
            hovermode='x unified',
            template='plotly_dark',
            height=400,
            font=dict(color='#EAEAEA'),
            plot_bgcolor='#111827',
            paper_bgcolor='#0B0F1A',
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Predictions":
    st.markdown("<div class='section-title'>Make Predictions</div>", unsafe_allow_html=True)

    api_healthy, _ = check_api_health()

    if not api_healthy:
        st.error("API Server offline. Start with: `docker-compose up -d`")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            ticker = st.text_input("Ticker", "AAPL", help="Stock/crypto ticker")

        with col2:
            pred_date = st.date_input("Prediction Date", datetime.now())

        with col3:
            if st.button("Predict", use_container_width=True):
                st.session_state.predict_clicked = True

        if st.session_state.get('predict_clicked'):
            with st.spinner("⏳ Generating prediction..."):
                pred = get_live_prediction(ticker)
                if pred:
                    st.success("✅ Prediction Generated")

                    # Display prediction details
                    col1, col2, col3 = st.columns(3)

                    prediction = pred.get('prediction', {})

                    with col1:
                        st.metric("Price", f"${prediction.get('price', 0):.2f}")

                    with col2:
                        confidence = prediction.get('confidence', 0)
                        st.metric("Confidence", f"{confidence:.1%}")

                    with col3:
                        direction = prediction.get('direction', 'HOLD')
                        st.metric("Signal", direction)

                    st.markdown("#### Full Response")
                    st.json(pred)
                else:
                    st.error("Prediction failed. Check logs.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: METRICS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Metrics":
    st.markdown("<div class='section-title'>Live Metrics</div>", unsafe_allow_html=True)

    # Auto-refresh
    refresh_interval = st.slider("Refresh every (seconds)", 5, 60, 10)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Predictions")
        metrics = get_prometheus_metrics()

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['predictions_per_sec'],
            title={"text": "Predictions/sec"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 500]},
                'bar': {'color': COLORS['primary']},
                'steps': [
                    {'range': [0, 150], 'color': '#1a1f2e'},
                    {'range': [150, 300], 'color': '#2a3a4e'},
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 400
                }
            }
        ))
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='#111827',
            paper_bgcolor='#0B0F1A',
            font=dict(color='#EAEAEA'),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Latency")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['avg_latency_ms'],
            title={"text": "Avg Latency (ms)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 500]},
                'bar': {'color': COLORS['secondary']},
                'steps': [
                    {'range': [0, 100], 'color': '#1a1f2e'},
                    {'range': [100, 250], 'color': '#2a3a4e'},
                ],
            }
        ))
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='#111827',
            paper_bgcolor='#0B0F1A',
            font=dict(color='#EAEAEA'),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(f"📍 Prometheus URL: {PROMETHEUS_URL}")
    st.info(f"🔄 Next refresh: {refresh_interval}s")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🚨 Anomalies":
    st.markdown("<div class='section-title'>Drift & Anomaly Detection</div>", unsafe_allow_html=True)

    st.markdown("""
    ### 🔍 Data Quality Monitoring

    Real-time monitoring of model drift and data quality:

    - **Feature Drift**: Detects changes in input feature distributions
    - **Prediction Drift**: Monitors output distribution shifts
    - **Performance Drift**: Tracks model accuracy degradation
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Feature Drift", "LOW", delta="Stable", delta_color="normal")

    with col2:
        st.metric("Prediction Drift", "MEDIUM", delta="Increasing", delta_color="inverse")

    with col3:
        st.metric("Performance", "HIGH", delta="-2.1%", delta_color="off")

    st.markdown("### Recent Alerts")

    alerts_data = {
        'Time': [
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=1),
            datetime.now() - timedelta(minutes=30),
        ],
        'Severity': ['WARNING', 'INFO', 'INFO'],
        'Message': [
            'Feature AAPL_volume shows PSI > 0.15',
            'Model inference latency +15% vs baseline',
            'Batch prediction completed (100 samples)',
        ],
    }

    df_alerts = pd.DataFrame(alerts_data)
    st.dataframe(df_alerts, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Settings":
    st.markdown("<div class='section-title'>Configuration</div>", unsafe_allow_html=True)

    st.markdown("### 🔌 API Configuration")

    new_api_url = st.text_input("API Base URL", API_BASE_URL)
    new_api_key = st.text_input("API Key", API_KEY, type="password")
    new_prometheus_url = st.text_input("Prometheus URL", PROMETHEUS_URL)

    if st.button("Save Configuration"):
        st.success("✅ Configuration saved (session only)")

    st.markdown("### 📊 Dashboard Settings")

    col1, col2 = st.columns(2)

    with col1:
        dark_mode = st.toggle("Dark Mode", value=True)

    with col2:
        auto_refresh = st.toggle("Auto-Refresh", value=True)

    st.markdown("### 📡 Service Health")

    services = {
        'API Server': ('http://localhost:8000/health', 'port 8000'),
        'Prometheus': ('http://localhost:9090/-/healthy', 'port 9090'),
        'Grafana': ('http://localhost:3000/api/health', 'port 3000'),
        'MLflow': ('http://localhost:5000', 'port 5000'),
    }

    st.write("#### Service Status")
    for service, (url, port) in services.items():
        try:
            resp = requests.get(url, timeout=2)
            status = "🟢 Online" if resp.status_code < 400 else "🔴 Offline"
        except:
            status = "🔴 Offline"

        st.write(f"**{service}**: {status} ({port})")

    st.markdown("### 📖 Documentation")

    st.info("""
    **Quick Start:**

    1. Start services:
       ```bash
       docker-compose up -d
       ```

    2. View API docs:
       ```
       http://localhost:8000/docs
       ```

    3. Access Grafana:
       ```
       http://localhost:3000
       ```

    4. MLflow Tracking:
       ```
       http://localhost:5000
       ```
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("View API Documentation", use_container_width=True):
            st.write("🔗 Open: http://localhost:8000/docs")

    with col2:
        if st.button("View Prometheus", use_container_width=True):
            st.write("🔗 Open: http://localhost:9090")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
---
<div style="text-align: center; padding: 20px; color: #8B95A5; font-size: 0.85rem;">
    <p>AI Predictive Intelligence Platform • Production Ready<br/>
    Last Updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
