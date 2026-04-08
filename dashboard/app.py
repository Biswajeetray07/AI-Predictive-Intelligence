"""
AI Predictive Intelligence — Real-Time Command Center Dashboard
================================================================
Enterprise-grade real-time analytics dashboard built with Streamlit.
Connects to FastAPI REST API for live predictions and monitoring.

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import sys
import datetime
import glob
import psutil
import requests
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.styles import MAIN_CSS, PLOTLY_TEMPLATE, COLORS
from dashboard.utils import (
    get_overview_kpis,
    get_data_sources_info,
    get_datasets_info,
    get_model_info,
    get_system_stats,
    get_pipeline_stages,
    load_stock_data,
    list_available_tickers,
    get_recent_logs,
    get_project_root,
    get_disk_usage_summary,
    get_training_history,
    get_pipeline_run_history,
    load_regime_states,
    get_regime_label,
    load_social_signals,
    load_macro_signals,
    load_crypto_data,
    list_crypto_coins,
    load_nlp_signals,
    load_nlp_label_quality,
    load_alternative_data_index,
    get_alternative_data_summary,
    load_technical_indicators,
    run_prediction_for_ticker,
    load_test_metadata,
    load_economic_indicators,
    load_weather_data,
    load_energy_data,
    load_news_articles,
    load_research_papers,
    load_patent_data,
    load_jobs_data,
    load_trade_data,
    load_population_data,
    REGIME_LABELS,
)

# Real-time API Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000") if "API_BASE_URL" in st.secrets else "http://localhost:8000"
API_KEY = st.secrets.get("API_KEY", "predictive_intel_dev_key_2026") if "API_KEY" in st.secrets else "predictive_intel_dev_key_2026"
PROMETHEUS_URL = st.secrets.get("PROMETHEUS_URL", "http://localhost:9090") if "PROMETHEUS_URL" in st.secrets else "http://localhost:9090"

# ═══════════════════════════════════════════════════════════════════════════════
# APP CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Predictive Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(MAIN_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# REAL-TIME API HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def get_api_health() -> Dict:
    """Check API health status."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3, headers={"X-API-Key": API_KEY})
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return {"status": "offline", "version": "N/A", "timestamp": "N/A"}

@st.cache_data(ttl=10)
def get_prometheus_metrics() -> Dict:
    """Fetch metrics from Prometheus."""
    metrics = {
        "predictions_total": 0,
        "predictions_per_sec": 0,
        "avg_latency_ms": 0,
        "error_rate": 0,
        "feature_drift": 0,
    }
    try:
        # Predictions per second (last 5 minutes)
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "rate(predictions_total[5m])"},
            timeout=3
        )
        if resp.status_code == 200:
            data = resp.json()
            if data['data']['result']:
                metrics['predictions_per_sec'] = float(data['data']['result'][0]['value'][1])
        
        # Average latency
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "avg(prediction_latency_seconds)"},
            timeout=3
        )
        if resp.status_code == 200:
            data = resp.json()
            if data['data']['result']:
                metrics['avg_latency_ms'] = float(data['data']['result'][0]['value'][1]) * 1000
        
        # Error rate
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "rate(prediction_errors_total[5m])"},
            timeout=3
        )
        if resp.status_code == 200:
            data = resp.json()
            if data['data']['result']:
                metrics['error_rate'] = float(data['data']['result'][0]['value'][1]) * 100
    except:
        pass
    return metrics

def get_live_prediction(ticker: str, date: str = None) -> Optional[Dict]:
    """Get live prediction from API."""
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
    except:
        pass
    return None

def get_batch_predictions(tickers: List[str], dates: List[str] = None) -> Optional[Dict]:
    """Get batch predictions from API."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/predictions/batch",
            params={"tickers": ",".join(tickers), "dates": ",".join(dates or [])},
            timeout=15,
            headers={"X-API-Key": API_KEY}
        )
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def get_api_metrics() -> Optional[Dict]:
    """Get comprehensive metrics from API endpoint."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/metrics",
            timeout=5,
            headers={"X-API-Key": API_KEY}
        )
        if resp.status_code == 200:
            return resp.text  # Raw Prometheus format
    except:
        pass
    return None

@st.cache_data(ttl=60)
def get_api_info() -> Dict:
    """Get API info including available models and data sources."""
    info = {
        "models": [],
        "data_sources": [],
        "version": "N/A",
    }
    try:
        # Get models info
        resp = requests.get(f"{API_BASE_URL}/info/models", timeout=5, headers={"X-API-Key": API_KEY})
        if resp.status_code == 200:
            info["models"] = resp.json().get("models", [])
        
        # Get data sources info
        resp = requests.get(f"{API_BASE_URL}/info/data-sources", timeout=5, headers={"X-API-Key": API_KEY})
        if resp.status_code == 200:
            info["data_sources"] = resp.json().get("sources", [])
    except:
        pass
    return info

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <div style="font-size: 2.5rem;">❖</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #f4f4f5; letter-spacing: -0.02em; margin-top: 8px;">
            Predictive AI
        </div>
        <div style="font-size: 0.75rem; color: #a1a1aa; font-weight: 500;">
            Command Center
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height: 1px; background: #27272a; margin: 16px 0;"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📡 Data Sources",
            "⚙️ Data Pipeline",
            "📊 Datasets",
            "🤖 Model Training",
            "📈 Model Evaluation",
            "🔮 Predictions",
            "🚨 Anomaly & Regimes",
            "📱 Social & NLP",
            "🌍 Alternative Data",
            "📉 Visualization",
            "💻 System Monitor",
            "📋 Logs",
            "🛠 Settings",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div style="height: 1px; background: #27272a; margin: 16px 0;"></div>', unsafe_allow_html=True)

    # System Status footer in sidebar
    try:
        stats = get_system_stats()
        st.markdown(f"""
        <div style="padding: 12px; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid #27272a;">
            <div style="font-size: 0.75rem; color: #f4f4f5; font-weight: 500; margin-bottom: 8px;">
                Systems
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #a1a1aa; font-size: 0.75rem;">CPU</span>
                <span style="color: {'#ef4444' if stats['cpu_percent'] > 80 else '#10b981'}; font-size: 0.75rem; font-family: 'Geist Mono';">
                    {stats['cpu_percent']}%
                </span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #a1a1aa; font-size: 0.75rem;">RAM</span>
                <span style="color: {'#ef4444' if stats['memory_percent'] > 80 else '#3b82f6'}; font-size: 0.75rem; font-family: 'Geist Mono';">
                    {stats['memory_percent']}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # S3 Status indicator
        from dashboard.utils import USE_S3, S3_BUCKET
        if USE_S3:
            s3_status_color = '#10b981' # green
            s3_status_text = 'Connected'
        else:
            s3_status_color = '#a1a1aa' # grey
            s3_status_text = 'Local Only'
            
        st.markdown(f"""
        <div style="padding: 12px; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid #27272a; margin-top: 12px;">
            <div style="font-size: 0.75rem; color: #f4f4f5; font-weight: 500; margin-bottom: 8px;">
                Storage Mode
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #a1a1aa; font-size: 0.75rem;">Backend</span>
                <span style="color: {s3_status_color}; font-size: 0.75rem; font-family: 'Geist Mono';">
                    ● {s3_status_text}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #a1a1aa; font-size: 0.75rem;">Bucket</span>
                <span style="color: #3b82f6; font-size: 0.65rem; font-family: 'Geist Mono'; text-align:right">
                    {S3_BUCKET if USE_S3 else 'N/A'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def render_section_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 600; color: #f8fafc;">{title}</h2>
        {f'<p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

def page_overview():
    # ── Load Real Data ──
    kpis = get_overview_kpis()
    stats = get_system_stats()
    disk = get_disk_usage_summary()
    jobs = get_pipeline_run_history()
    models = get_model_info()
    stages = get_pipeline_stages()

    completed = sum(1 for s in stages if s['status'] == 'Complete')
    not_started = sum(1 for s in stages if s['status'] == 'Not Started')
    empty = sum(1 for s in stages if s['status'] == 'Empty')

    # ── Header Row ──
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown('<div style="font-size: 1.8rem; font-weight: 600; margin-bottom: 4px;">AI Predictive Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div style="color: #a1a1aa; font-size: 0.85rem; margin-bottom: 20px;">End-to-end ML platform that collects 15+ real-world data domains, trains deep learning ensembles (LSTM, GRU, Transformer, TFT), and generates multi-horizon stock price forecasts enriched with macro, energy, social, and NLP signals.</div>', unsafe_allow_html=True)
    with header_col2:
        api_count = kpis.get('active_apis', 0)
        st.markdown(f'<div style="text-align: right; color: #a1a1aa; font-size: 0.9rem; margin-top: 10px;">🟢 APIs Active: {api_count} &nbsp;&nbsp; 📊 Records: {kpis.get("total_records", 0):,}</div>', unsafe_allow_html=True)

    # ── ROW 1 (3 Cards) ──
    r1c1, r1c2, r1c3 = st.columns(3)

    # Mini bar chart helper using real breakdowns
    def draw_mini_bar(labels, values, color):
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color=color, width=0.4))
        fig.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig.update_layout(height=140, margin=dict(l=0,r=0,t=10,b=10),
            xaxis=dict(showgrid=False, zeroline=False, color='#a1a1aa'),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        return fig

    with r1c1:
        st.markdown(f"""
        <div class="bento-card">
            <div class="card-title">📡 Data Sources</div>
            <div class="card-subtitle">Live data from pipeline collection layer</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                <div><span style="font-size:0.8rem;color:#a1a1aa;">Sources</span><br/><span style="font-size:1.5rem;font-weight:600;">{kpis.get('data_sources', 0)}</span></div>
                <div><span style="font-size:0.8rem;color:#a1a1aa;">Features</span><br/><span style="font-size:1.5rem;font-weight:600;">{kpis.get('features_generated', 0)}</span></div>
            </div>
        """, unsafe_allow_html=True)
        # Mini chart from disk breakdown
        bd = disk['breakdown']
        st.plotly_chart(draw_mini_bar(list(bd.keys()), list(bd.values()), '#a7f3d0'), use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown(f"""
        <div class="bento-card">
            <div class="card-title">🤖 Models</div>
            <div class="card-subtitle">Trained model artifacts and checkpoints</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                <div><span style="font-size:0.8rem;color:#a1a1aa;">Saved Models</span><br/><span style="font-size:1.5rem;font-weight:600;">{kpis.get('models_saved', 0)}</span></div>
                <div><span style="font-size:0.8rem;color:#a1a1aa;">Total Size</span><br/><span style="font-size:1.5rem;font-weight:600;">{disk['breakdown'].get('Models', 0)} MB</span></div>
            </div>
        """, unsafe_allow_html=True)
        if models:
            names = [m['name'][:8] for m in models[:6]]
            sizes = [m['size_mb'] for m in models[:6]]
            st.plotly_chart(draw_mini_bar(names, sizes, '#3b82f6'), use_container_width=True, config={'displayModeBar':False})
        else:
            st.caption('No models trained yet')
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c3:
        st.markdown(f"""
        <div class="bento-card">
            <div class="card-title">⚙️ Pipeline Stages</div>
            <div class="card-subtitle">Real-time pipeline execution status</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                <div><span style="font-size:0.8rem;color:#10b981;">■ Complete</span><br/><span style="font-size:1.5rem;font-weight:600;">{completed}</span></div>
                <div><span style="font-size:0.8rem;color:#f59e0b;">■ Empty</span><br/><span style="font-size:1.5rem;font-weight:600;">{empty}</span></div>
                <div><span style="font-size:0.8rem;color:#a1a1aa;">■ Pending</span><br/><span style="font-size:1.5rem;font-weight:600;">{not_started}</span></div>
            </div>
        """, unsafe_allow_html=True)
        stage_names = [s['name'][:5] for s in stages]
        stage_files = [s['files'] for s in stages]
        fig_stages = go.Figure(go.Bar(x=stage_names, y=stage_files, marker_color=['#10b981' if s['status']=='Complete' else '#f59e0b' if s['status']=='Empty' else '#27272a' for s in stages], width=0.4))
        fig_stages.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig_stages.update_layout(height=140, margin=dict(l=0,r=0,t=10,b=10),
            barmode='stack', showlegend=False, xaxis=dict(showgrid=False, showticklabels=True, tickfont=dict(size=9)),
            yaxis=dict(showgrid=False, showticklabels=False))
        st.plotly_chart(fig_stages, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ROWS 2 & 3 (Vertical Panel Layout) ──
    left_block, right_panel = st.columns([2, 1])

    with left_block:
        # Row 2 — System Resources (LIVE)
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            cpu_color = '#ef4444' if stats['cpu_percent'] > 80 else '#f59e0b' if stats['cpu_percent'] > 50 else '#10b981'
            st.markdown(f"""
            <div class="bento-card">
                <div class="card-title">🖥 CPU & Memory</div>
                <div class="card-subtitle">Live system resource utilization</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                    <div><span style="font-weight:600;font-size:1.1rem;color:{cpu_color};">{stats['cpu_percent']}%</span> <span style="font-size:0.7rem;color:#a1a1aa;">CPU</span></div>
                    <div><span style="font-weight:600;font-size:1.1rem;">{stats['memory_percent']}%</span> <span style="font-size:0.7rem;color:#a1a1aa;">RAM</span></div>
                    <div><span style="font-weight:600;font-size:1.1rem;">{stats['memory_used_gb']}GB</span> <span style="font-size:0.7rem;color:#a1a1aa;">Used</span></div>
                </div>
            """, unsafe_allow_html=True)
            # CPU gauge mini chart
            fig_cpu = go.Figure(go.Indicator(mode="gauge+number", value=stats['cpu_percent'],
                number=dict(suffix="%", font=dict(color=cpu_color, size=20)),
                gauge=dict(axis=dict(range=[0,100], visible=False), bar=dict(color=cpu_color),
                    bgcolor='#18181b', borderwidth=0)))
            fig_cpu.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=100, margin=dict(l=20,r=20,t=10,b=10))
            st.plotly_chart(fig_cpu, use_container_width=True, config={'displayModeBar':False})
            st.markdown('</div>', unsafe_allow_html=True)

        with r2c2:
            st.markdown(f"""
            <div class="bento-card">
                <div class="card-title">💾 Disk Usage</div>
                <div class="card-subtitle">Storage footprint of pipeline data</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                    <div><span style="font-weight:600;font-size:1.1rem;">{stats['disk_percent']}%</span> <span style="font-size:0.7rem;color:#a1a1aa;">Disk</span></div>
                    <div><span style="font-weight:600;font-size:1.1rem;">{stats['disk_used_gb']}GB</span> <span style="font-size:0.7rem;color:#a1a1aa;">Used</span></div>
                    <div><span style="font-weight:600;font-size:1.1rem;">{stats['disk_total_gb']}GB</span> <span style="font-size:0.7rem;color:#a1a1aa;">Total</span></div>
                </div>
            """, unsafe_allow_html=True)
            fig_disk = go.Figure(go.Indicator(mode="gauge+number", value=stats['disk_percent'],
                number=dict(suffix="%", font=dict(color='#3b82f6', size=20)),
                gauge=dict(axis=dict(range=[0,100], visible=False), bar=dict(color='#3b82f6'),
                    bgcolor='#18181b', borderwidth=0)))
            fig_disk.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=100, margin=dict(l=20,r=20,t=10,b=10))
            st.plotly_chart(fig_disk, use_container_width=True, config={'displayModeBar':False})
            st.markdown('</div>', unsafe_allow_html=True)

        # Row 3 — Real Data Summary
        st.markdown('<div class="card-title" style="margin-bottom:4px;margin-top:8px;">📊 Collected Data Summary</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#71717a;font-size:0.75rem;margin-bottom:10px;">All data domains below are actively collected and fed as features into the AI prediction models. More data sources → richer context → better forecasts.</div>', unsafe_allow_html=True)
        tickers = list_available_tickers()
        ticker_count = len(tickers) if tickers else 0
        crypto_df = load_crypto_data()
        crypto_count = crypto_df['coin'].nunique() if crypto_df is not None and 'coin' in crypto_df.columns else 0
        social_df = load_social_signals()
        social_days = len(social_df) if social_df is not None else 0
        fred_df = load_economic_indicators()
        fred_count = fred_df['indicator'].nunique() if fred_df is not None and 'indicator' in fred_df.columns else 0
        weather_df = load_weather_data()
        weather_count = len(weather_df) if weather_df is not None else 0
        energy_df = load_energy_data()
        energy_series = energy_df['series'].nunique() if energy_df is not None and 'series' in energy_df.columns else 0
        research_df = load_research_papers()
        research_count = len(research_df) if research_df is not None else 0
        jobs_df_ov = load_jobs_data()
        jobs_count = len(jobs_df_ov) if jobs_df_ov is not None else 0

        data_items = [
            ('📈 Stocks', f'{ticker_count} tickers', '10yr OHLCV + 17 tech indicators'),
            ('🪙 Crypto', f'{crypto_count} coins', 'Price, volume, market cap'),
            ('💬 Social', f'{social_days:,} days', 'GitHub, HN, YT, Mastodon signals'),
            ('🏛 Economy', f'{fred_count} indicators', 'FRED: GDP, CPI, VIX, rates'),
            ('⚡ Energy', f'{energy_series} series', 'EIA: Oil, Gas, Coal, Electricity'),
            ('🌤 Weather', f'{weather_count} cities', 'Global temperature & humidity'),
            ('📄 Research', f'{research_count:,} papers', 'arXiv AI/ML/Finance papers'),
            ('💼 Jobs', f'{jobs_count:,} listings', 'Adzuna + USAJobs market data'),
        ]

        d_cols = st.columns(4)
        for idx, (icon, val, desc) in enumerate(data_items[:4]):
            with d_cols[idx]:
                st.markdown(f'''<div class="bento-card" style="padding:10px;text-align:center;">
                    <div style="font-size:0.7rem;color:#a1a1aa;">{icon}</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#f4f4f5;">{val}</div>
                    <div style="font-size:0.6rem;color:#71717a;">{desc}</div>
                </div>''', unsafe_allow_html=True)
        d_cols2 = st.columns(4)
        for idx, (icon, val, desc) in enumerate(data_items[4:]):
            with d_cols2[idx]:
                st.markdown(f'''<div class="bento-card" style="padding:10px;text-align:center;">
                    <div style="font-size:0.7rem;color:#a1a1aa;">{icon}</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#f4f4f5;">{val}</div>
                    <div style="font-size:0.6rem;color:#71717a;">{desc}</div>
                </div>''', unsafe_allow_html=True)

    with right_panel:
        # Build jobs HTML dynamically from real pipeline data
        jobs_html = ''
        for job in jobs:
            badge_class = 'badge-active' if job['status'] == 'Succeeded' else 'badge-warning' if job['status'] == 'Empty' else 'badge-neutral'
            jobs_html += f"""<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
<span style="color:#f4f4f5; font-size:0.85rem; background:#27272a; padding:4px 8px; border-radius:4px;">{job['name']}</span>
<span class="badge {badge_class}"><div class='badge-dot' style='background-color:{job['color']};'></div>{job['status']}</span>
<span style="color:#a1a1aa; font-size:0.75rem;">{job['time']}</span>
</div>"""

        if not jobs_html:
            jobs_html = '<div style="color:#a1a1aa; text-align:center; padding:20px;">No pipeline stages detected yet. Run the pipeline to see status here.</div>'

        st.markdown(f"""
        <div class="bento-card" style="height: 100%;">
            <div class="card-title">⏱ Pipeline Status</div>
            <div class="card-subtitle">{completed}/{len(stages)} stages complete · {sum(s['files'] for s in stages)} total files</div>
            <div style="display:flex; flex-direction:column; gap:16px; margin-top:20px;">
                {jobs_html}
            </div>
        </div>
        """, unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

def page_data_sources():
    render_section_header("Data Sources", "Monitor all API data collection sources")

    st.markdown('''
    <div style="color:#71717a;font-size:0.78rem;margin-bottom:16px;line-height:1.5;">
        Each card below represents a live API data source that the platform connects to. Data is collected via automated scripts, stored in <code style="color:#3b82f6;">data/raw/</code>,
        then processed into feature matrices for model training. Green “Active” badges indicate the source has data; the platform tracks file counts, record volumes, and last-update timestamps.
    </div>
    ''', unsafe_allow_html=True)

    sources = get_data_sources_info()
    if not sources:
        st.warning("No data sources detected. Run the data collection pipeline first.")
        return

    # ── Source Cards Grid ──
    cols = st.columns(3)
    for i, s in enumerate(sources):
        with cols[i % 3]:
            badge_type = "badge-active" if s['status'] in ('Online', 'Complete', 'Active') else "badge-warning"
            badge_dot = "<div class='badge-dot'></div>"
            
            st.markdown(f"""
            <div class="bento-card">
                <div class="card-header">
                    <h3 class="card-title">{s['name']}</h3>
                    <span class="badge {badge_type}">{badge_dot}{s['status']}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;">
                    <div>
                        <div style="color: #a1a1aa; font-size: 0.7rem; font-weight: 500;">FILES</div>
                        <div style="color: #f4f4f5; font-size: 1.1rem; font-weight: 700; font-family: 'Geist Mono';">{s['files']}</div>
                    </div>
                    <div>
                        <div style="color: #a1a1aa; font-size: 0.7rem; font-weight: 500;">RECORDS</div>
                        <div style="color: #f4f4f5; font-size: 1.1rem; font-weight: 700; font-family: 'Geist Mono';">{s['records']:,}</div>
                    </div>
                    <div>
                        <div style="color: #a1a1aa; font-size: 0.7rem; font-weight: 500;">SIZE</div>
                        <div style="color: #a1a1aa; font-size: 0.9rem; font-family: 'Geist Mono';">{s['size_mb']} MB</div>
                    </div>
                    <div>
                        <div style="color: #a1a1aa; font-size: 0.7rem; font-weight: 500;">LAST UPDATE</div>
                        <div style="color: #a1a1aa; font-size: 0.8rem;">{s['last_updated']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)

    # ── Aggregate Charts ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card-title" style="margin-bottom: 16px;">Data Volume by Source</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=[s['name'] for s in sources],
            values=[s['records'] for s in sources],
            hole=0.6,
            marker=dict(colors=['#f4f4f5', '#a1a1aa', '#3b82f6', '#10b981', '#f59e0b', '#ef4444']),
            textinfo='label+percent',
            textposition='outside',
        ))
        fig.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig.update_layout( height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="card-title" style="margin-bottom: 16px;">Storage footprint (MB)</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=[s['name'] for s in sources],
            y=[s['size_mb'] for s in sources],
            marker_color='#3b82f6',
            opacity=0.8,
            text=[f"{s['size_mb']}MB" for s in sources],
            textposition='auto',
        ))
        fig.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig.update_layout( height=400)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def page_pipeline():
    render_section_header("Data Pipeline", "End-to-end ML pipeline architecture and data flow")

    st.markdown('''
    <div style="color:#71717a;font-size:0.78rem;margin-bottom:16px;line-height:1.5;">
        The pipeline processes raw market data through 5 stages: <strong style="color:#a1a1aa;">Raw Data</strong> (API collection) →
        <strong style="color:#a1a1aa;">Processed</strong> (cleaning + technical indicators) →
        <strong style="color:#a1a1aa;">Features</strong> (feature engineering) →
        <strong style="color:#a1a1aa;">Model Inputs</strong> (numpy arrays + train/test splits) →
        <strong style="color:#a1a1aa;">Models</strong> (trained neural networks). Each stage builds on the previous one.
    </div>
    ''', unsafe_allow_html=True)

    stages = get_pipeline_stages()
    disk = get_disk_usage_summary()

    # ── Pipeline Summary KPIs ──
    total_files = sum(s['files'] for s in stages)
    completed = sum(1 for s in stages if s['status'] == 'Complete')

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="bento-card" style="padding:12px;text-align:center;border-top:3px solid #10b981;"><div style="color:#a1a1aa;font-size:0.7rem;">STAGES COMPLETE</div><div style="font-size:1.4rem;font-weight:700;color:#10b981;">{completed}/{len(stages)}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="bento-card" style="padding:12px;text-align:center;border-top:3px solid #3b82f6;"><div style="color:#a1a1aa;font-size:0.7rem;">TOTAL FILES</div><div style="font-size:1.4rem;font-weight:700;color:#3b82f6;">{total_files:,}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="bento-card" style="padding:12px;text-align:center;border-top:3px solid #f59e0b;"><div style="color:#a1a1aa;font-size:0.7rem;">DISK USAGE</div><div style="font-size:1.4rem;font-weight:700;color:#f59e0b;">{disk["total_gb"]} GB</div></div>', unsafe_allow_html=True)
    with k4:
        kpis = get_overview_kpis()
        st.markdown(f'<div class="bento-card" style="padding:12px;text-align:center;border-top:3px solid #a855f7;"><div style="color:#a1a1aa;font-size:0.7rem;">TOTAL RECORDS</div><div style="font-size:1.4rem;font-weight:700;color:#a855f7;">{kpis.get("total_records", 0):,}</div></div>', unsafe_allow_html=True)

    # ── Visual Pipeline Flow ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 24px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:16px;">🔄 Pipeline Execution Flow</div>', unsafe_allow_html=True)

    pipeline_desc = {
        'Raw Data': '502 stock tickers, 100 crypto coins, social signals, macro indicators',
        'Processed': 'Cleaned OHLCV with 17 technical indicators per ticker (parquet)',
        'Features': 'Feature-engineered datasets for model training',
        'Model Inputs': 'X_train/X_test numpy arrays, labels, metadata',
        'Models': '14 trained models (LSTM, GRU, Transformer, TFT, NLP, Fusion, HMM)',
    }

    flow_html = '<div style="display:flex;gap:8px;align-items:center;overflow-x:auto;padding:8px 0;">'
    for i, s in enumerate(stages):
        color = '#10b981' if s['status'] == 'Complete' else '#f59e0b' if s['status'] == 'Empty' else '#27272a'
        desc = pipeline_desc.get(s['name'], '')
        flow_html += f'''<div class="bento-card" style="min-width:180px;padding:12px;text-align:center;border-top:3px solid {color};">
            <div style="font-weight:600;color:#f4f4f5;font-size:0.85rem;">{s['name']}</div>
            <div style="color:{color};font-size:0.7rem;margin:4px 0;">● {s['status']}</div>
            <div style="font-family:'Geist Mono';color:#3b82f6;font-size:0.9rem;font-weight:700;">{s['files']:,} files</div>
            <div style="color:#71717a;font-size:0.6rem;margin-top:4px;">{desc}</div></div>'''
        if i < len(stages) - 1:
            flow_html += '<div style="color:#27272a;font-size:1.2rem;">→</div>'
    flow_html += '</div>'
    st.markdown(flow_html, unsafe_allow_html=True)

    # ── Data Breakdown Chart ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 24px 0;"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card-title" style="margin-bottom:12px;">Files per Stage</div>', unsafe_allow_html=True)
        fig_bar = go.Figure(go.Bar(
            x=[s['name'] for s in stages], y=[s['files'] for s in stages],
            marker_color=['#10b981' if s['status'] == 'Complete' else '#f59e0b' for s in stages],
            text=[f"{s['files']:,}" for s in stages], textposition='auto'))
        fig_bar.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig_bar.update_layout(height=300)
        st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        st.markdown('<div class="card-title" style="margin-bottom:12px;">Disk Usage by Category</div>', unsafe_allow_html=True)
        bd = disk['breakdown']
        fig_pie = go.Figure(go.Pie(labels=list(bd.keys()), values=list(bd.values()), hole=0.5,
            marker=dict(colors=['#3b82f6', '#10b981', '#f59e0b', '#a855f7', '#ef4444'][:len(bd)]),
            textinfo='label+value'))
        fig_pie.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig_pie.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Stage Details Table ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 24px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:12px;">Stage Manifest</div>', unsafe_allow_html=True)
    df = pd.DataFrame(stages)
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={
                     'name': st.column_config.TextColumn('Stage'),
                     'status': st.column_config.TextColumn('Status'),
                     'files': st.column_config.NumberColumn('Files'),
                     'last_modified': st.column_config.TextColumn('Last Modified'),
                 })


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

def page_datasets():
    render_section_header("Datasets", "Explore processed datasets and feature tables")

    datasets = get_datasets_info()
    if not datasets:
        st.warning("No processed datasets found. Run the data processing pipeline first.")
        return

    # ── Dataset Cards ──
    cols = st.columns(3)
    for i, d in enumerate(datasets[:12]):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="bento-card">
                <div class="card-header">
                    <h3 class="card-title">📄 {d['name']}</h3>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px;">
                    <div>
                        <div style="color: #a1a1aa; font-size: 0.7rem; font-weight: 500;">RECORDS</div>
                        <div style="color: #f4f4f5; font-size: 1.1rem; font-weight: 700; font-family: 'Geist Mono';">{d['records']:,}</div>
                    </div>
                    <div>
                        <div style="color: #a1a1aa; font-size: 0.7rem; font-weight: 500;">FEATURES</div>
                        <div style="color: #3b82f6; font-size: 1.1rem; font-weight: 700; font-family: 'Geist Mono';">{d['features']}</div>
                    </div>
                </div>
                <div style="color: #a1a1aa; font-size: 0.8rem; margin-top: 12px; font-family: 'Geist Mono';">Size: {d['size_mb']} MB</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)

    # ── Dataset Explorer ──
    st.markdown('<div class="card-title" style="margin-bottom: 16px;">Dataset Explorer</div>', unsafe_allow_html=True)
    dataset_names = [d['name'] for d in datasets]
    selected = st.selectbox("Select a dataset to explore", dataset_names)
    sel_ds = next((d for d in datasets if d['name'] == selected), None)

    if sel_ds:
        st.caption(f"**Columns ({sel_ds['features']}):** `{'`, `'.join(sel_ds['columns'][:20])}`{'...' if len(sel_ds['columns']) > 20 else ''}")
        try:
            if sel_ds['path'].endswith('.csv'):
                preview = pd.read_csv(sel_ds['path'], nrows=100)
            else:
                preview = pd.read_parquet(sel_ds['path']).head(100)
            st.dataframe(preview, use_container_width=True, height=400)
        except Exception as e:
            st.error(f"Could not load preview: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def page_model_training():
    render_section_header("Model Training Lab", "Track experiments, losses, and model artifacts")

    models = get_model_info()

    if not models:
        st.info("No saved models found yet. Training may still be in progress...")
        st.markdown("""
        <div class="bento-card" style="text-align: center; padding: 40px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px; color: #a1a1aa;">⏳</div>
            <h3 class="card-title">Training in Progress</h3>
            <p style="color: #a1a1aa; font-size: 0.9rem;">Models will appear here once training completes.<br>
            Check System Monitor for CPU/RAM usage to confirm training is active.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Model Cards ──
    st.markdown('<div class="card-title" style="margin-bottom: 16px;">Trained Models</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(models), 3))
    for i, m in enumerate(models):
        with cols[i % len(cols)]:
            type_color = '#3b82f6' if m['type'] == 'NLP Multi-Task' else '#10b981' if m['type'] == 'Time Series' else '#f59e0b'
            st.markdown(f"""
            <div class="bento-card">
                <div class="card-header">
                    <h3 class="card-title">{m['name']}</h3>
                    <span class="badge badge-neutral" style="color: {type_color}; border-color: {type_color}40;"><div class='badge-dot' style='background-color:{type_color};'></div>{m['type']}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px;">
                    <div>
                        <div style="color: #a1a1aa; font-size: 0.7rem; font-weight: 500;">ALGORITHM</div>
                        <div style="color: #f4f4f5; font-size: 0.9rem;">{m['algorithm']}</div>
                    </div>
                    <div>
                        <div style="color: #a1a1aa; font-size: 0.7rem; font-weight: 500;">SIZE</div>
                        <div style="color: #f4f4f5; font-size: 1rem; font-weight: 700; font-family: 'Geist Mono';">{m['size_mb']} MB</div>
                    </div>
                </div>
                <div style="color: #a1a1aa; font-size: 0.8rem; margin-top: 16px;">Trained: {m['trained_at']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="card-title" style="margin-bottom: 16px;">Model Size Comparison</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=[m['name'] for m in models],
        y=[m['size_mb'] for m in models],
        marker_color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'][:len(models)],
        text=[f"{m['size_mb']} MB" for m in models],
        textposition='auto',
    ))
    fig.update_layout(**PLOTLY_TEMPLATE['layout'])
    fig.update_layout( height=350, yaxis_title="Size (MB)")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def page_model_evaluation():
    render_section_header("Model Evaluation", "Real model performance, architecture, and training metrics")

    st.markdown('''
    <div style="background: rgba(59,130,246,0.06); border: 1px solid #27272a; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
        <div style="color:#a1a1aa; font-size: 0.78rem; line-height: 1.5;">
            📊 <strong style="color:#f4f4f5;">What This Page Shows:</strong> Architecture breakdown of all trained models, test dataset statistics,
            NLP label quality analysis, and training loss curves. The test data is held out from training — the models have never seen these samples,
            making the evaluation metrics unbiased estimates of real-world performance.
        </div>
    </div>
    ''', unsafe_allow_html=True)

    models = get_model_info()
    if not models:
        st.info("No models available for evaluation yet.")
        return

    # ── Model Architecture Summary ──
    st.markdown('<div class="card-title" style="margin-bottom:16px;">🏗 Model Architecture Summary</div>', unsafe_allow_html=True)
    ts_models = [m for m in models if m['type'] == 'Time Series']
    nlp_models = [m for m in models if 'NLP' in m['type']]
    fusion_models = [m for m in models if m['type'] == 'Fusion']
    regime_models = [m for m in models if m['type'] == 'Regime Detection']

    arch_html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;">'
    arch_html += f'<div class="bento-card" style="padding:12px;text-align:center;border-top:3px solid #3b82f6;"><div style="color:#3b82f6;font-weight:700;font-size:1.2rem;">{len(ts_models)}</div><div style="color:#a1a1aa;font-size:0.75rem;">Time Series Models</div><div style="color:#71717a;font-size:0.65rem;">LSTM, GRU, Transformer, TFT</div></div>'
    arch_html += f'<div class="bento-card" style="padding:12px;text-align:center;border-top:3px solid #ef4444;"><div style="color:#ef4444;font-weight:700;font-size:1.2rem;">{len(nlp_models)}</div><div style="color:#a1a1aa;font-size:0.75rem;">NLP Models</div><div style="color:#71717a;font-size:0.65rem;">DeBERTa-v3 Multi-Task</div></div>'
    arch_html += f'<div class="bento-card" style="padding:12px;text-align:center;border-top:3px solid #f59e0b;"><div style="color:#f59e0b;font-weight:700;font-size:1.2rem;">{len(fusion_models)}</div><div style="color:#a1a1aa;font-size:0.75rem;">Fusion Models</div><div style="color:#71717a;font-size:0.65rem;">Cross-Attention Multi-Horizon</div></div>'
    arch_html += f'<div class="bento-card" style="padding:12px;text-align:center;border-top:3px solid #10b981;"><div style="color:#10b981;font-weight:700;font-size:1.2rem;">{len(regime_models)}</div><div style="color:#a1a1aa;font-size:0.75rem;">Regime Models</div><div style="color:#71717a;font-size:0.65rem;">HMM 5-State Detector</div></div>'
    arch_html += '</div>'
    st.markdown(arch_html, unsafe_allow_html=True)

    # ── Test Data Metrics ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 24px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:16px;">📊 Test Dataset Statistics</div>', unsafe_allow_html=True)

    meta_path = os.path.join(PROJECT_ROOT, "data", "processed", "model_inputs", "metadata_test.csv")
    y_test_path = os.path.join(PROJECT_ROOT, "data", "processed", "model_inputs", "y_test.npy")
    y_multi_path = os.path.join(PROJECT_ROOT, "data", "processed", "model_inputs", "y_multi_test.npy")

    if os.path.exists(meta_path) and os.path.exists(y_test_path):
        meta = pd.read_csv(meta_path)
        y_test = np.load(y_test_path)
        y_multi = np.load(y_multi_path) if os.path.exists(y_multi_path) else None

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="bento-card" style="padding:12px;text-align:center;"><div style="color:#a1a1aa;font-size:0.7rem;">TEST SAMPLES</div><div style="font-size:1.3rem;font-weight:700;color:#f4f4f5;">{len(y_test):,}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="bento-card" style="padding:12px;text-align:center;"><div style="color:#a1a1aa;font-size:0.7rem;">TICKERS</div><div style="font-size:1.3rem;font-weight:700;color:#f4f4f5;">{meta["ticker"].nunique()}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="bento-card" style="padding:12px;text-align:center;"><div style="color:#a1a1aa;font-size:0.7rem;">DATE RANGE</div><div style="font-size:0.85rem;font-weight:600;color:#f4f4f5;">{meta["date"].min()[:10]} → {meta["date"].max()[:10]}</div></div>', unsafe_allow_html=True)
        with m4:
            horizons = f"{y_multi.shape[1]} horizons" if y_multi is not None else "1 horizon"
            st.markdown(f'<div class="bento-card" style="padding:12px;text-align:center;"><div style="color:#a1a1aa;font-size:0.7rem;">PREDICTION HORIZONS</div><div style="font-size:1.3rem;font-weight:700;color:#f4f4f5;">{horizons}</div></div>', unsafe_allow_html=True)

        # Target distribution
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card-title" style="margin-bottom:8px;">Target Distribution (y_test)</div>', unsafe_allow_html=True)
            fig_h = go.Figure(go.Histogram(x=y_test.flatten(), nbinsx=80, marker_color='#3b82f6'))
            fig_h.update_layout(**PLOTLY_TEMPLATE['layout'])
            fig_h.update_layout(height=280, xaxis_title="Target Value")
            st.plotly_chart(fig_h, use_container_width=True)
        with c2:
            if y_multi is not None:
                st.markdown('<div class="card-title" style="margin-bottom:8px;">Multi-Horizon Targets</div>', unsafe_allow_html=True)
                fig_m = go.Figure()
                for i, label in enumerate(['1-day', '5-day', '30-day']):
                    fig_m.add_trace(go.Histogram(x=y_multi[:, i], name=label, opacity=0.7, nbinsx=60))
                fig_m.update_layout(**PLOTLY_TEMPLATE['layout'])
                fig_m.update_layout(height=280, barmode='overlay')
                st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("Test data not found. Run the full pipeline to generate model inputs.")

    # ── NLP Label Quality ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 24px 0;"></div>', unsafe_allow_html=True)
    lq = load_nlp_label_quality()
    if lq:
        st.markdown('<div class="card-title" style="margin-bottom:16px;">🧠 NLP Label Quality Analysis</div>', unsafe_allow_html=True)
        for task_name, info_str in lq.items():
            try:
                import ast
                info = ast.literal_eval(info_str) if isinstance(info_str, str) else info_str
                dist = info.get('distribution', {})
                entropy = info.get('normalized_entropy', 0)
                balanced = info.get('is_balanced', False)
                bal_badge = '<span style="color:#10b981;">✓ Balanced</span>' if balanced else '<span style="color:#ef4444;">✗ Imbalanced</span>'
                st.markdown(f'<div class="bento-card" style="padding:12px;margin-bottom:8px;"><strong style="color:#3b82f6;">{task_name.title()}</strong> {bal_badge} <span style="color:#a1a1aa;font-size:0.8rem;margin-left:12px;">Entropy: {entropy:.3f}</span></div>', unsafe_allow_html=True)
                if dist:
                    fig_d = go.Figure(go.Bar(x=[f"Class {k}" for k in dist.keys()], y=list(dist.values()), marker_color='#a855f7', text=list(dist.values()), textposition='auto'))
                    fig_d.update_layout(**PLOTLY_TEMPLATE['layout'])
                    fig_d.update_layout(height=200)
                    st.plotly_chart(fig_d, use_container_width=True)
            except Exception:
                st.markdown(f'<div class="bento-card" style="padding:12px;margin-bottom:8px;"><strong style="color:#3b82f6;">{task_name.title()}</strong><pre style="color:#a1a1aa;font-size:0.7rem;">{info_str}</pre></div>', unsafe_allow_html=True)

    # ── Training History ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    history = get_training_history()
    if history:
        st.markdown('<div class="card-title" style="margin-bottom: 16px;">Training Loss Curves (Real)</div>', unsafe_allow_html=True)
        model_colors = {'lstm': '#3b82f6', 'gru': '#10b981', 'transformer': '#f59e0b', 'tft': '#a855f7', 'nlp': '#ef4444', 'fusion': '#06b6d4'}
        fig = go.Figure()
        for model_name, entries in history.items():
            color = model_colors.get(model_name.lower(), '#f4f4f5')
            epochs = [e['epoch'] for e in entries]
            if any('train_loss' in e for e in entries):
                fig.add_trace(go.Scatter(x=epochs, y=[e.get('train_loss') for e in entries], name=f'{model_name} (train)', line=dict(color=color, width=2)))
            if any('val_loss' in e for e in entries):
                fig.add_trace(go.Scatter(x=epochs, y=[e.get('val_loss') for e in entries], name=f'{model_name} (val)', line=dict(color=color, width=2, dash='dash')))
        fig.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig.update_layout(height=400, xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No training history found in log files. This populates during model training.')
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def page_predictions():
    render_section_header("Predictions & Intelligence", "Real AI model forecasts using trained ensemble")

    st.markdown('''
    <div style="background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(168,85,247,0.06)); border: 1px solid #27272a; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
        <div style="color: #f4f4f5; font-size: 0.9rem; font-weight: 600; margin-bottom: 6px;">🧠 How Our AI Predictions Work</div>
        <div style="color: #a1a1aa; font-size: 0.78rem; line-height: 1.5;">
            This page runs <strong style="color:#3b82f6;">real-time ensemble inference</strong> combining 4 deep learning models (LSTM, GRU, Transformer, TFT)
            trained on 10+ years of market data. Each model independently predicts future returns, and a
            <strong style="color:#a855f7;">regime-adaptive ensemble</strong> dynamically weights their outputs based on the current market regime
            (detected by a Hidden Markov Model). Predictions are further enriched with macro-economic signals (FRED),
            energy data (EIA), social sentiment, and NLP-derived event scores to provide multi-horizon forecasts (1-day, 5-day, 30-day).
        </div>
    </div>
    ''', unsafe_allow_html=True)

    tickers = list_available_tickers()
    if not tickers:
        st.warning("No stock data available. Run the financial data pipeline first.")
        return

    c1, c2 = st.columns([1, 3])
    with c1:
        selected_ticker = st.selectbox("Select Asset", tickers, index=tickers.index('AAPL') if 'AAPL' in tickers else 0)

    df = load_stock_data(selected_ticker)
    if df is None or df.empty:
        st.error(f"No data found for {selected_ticker}")
        return

    date_col = 'Date' if 'Date' in df.columns else 'date'

    # ── Price chart with technical indicators ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 24px 0;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title" style="margin-bottom: 16px;">{selected_ticker} — Price & Technical Indicators</div>', unsafe_allow_html=True)

    fig = go.Figure()
    if 'Close' in df.columns:
        fig.add_trace(go.Scatter(x=df[date_col], y=df['Close'], name='Close', line=dict(color='#f4f4f5', width=1.5)))
    if 'SMA_10' in df.columns:
        valid_sma = df[df['SMA_10'] > 0]
        fig.add_trace(go.Scatter(x=valid_sma[date_col], y=valid_sma['SMA_10'], name='SMA 10', line=dict(color='#3b82f6', width=1, dash='dot')))
    if 'SMA_50' in df.columns:
        valid_sma50 = df[df['SMA_50'] > 0]
        fig.add_trace(go.Scatter(x=valid_sma50[date_col], y=valid_sma50['SMA_50'], name='SMA 50', line=dict(color='#f59e0b', width=1, dash='dot')))
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        valid_bb = df[df['BB_Upper'] > 0]
        fig.add_trace(go.Scatter(x=valid_bb[date_col], y=valid_bb['BB_Upper'], name='BB Upper', line=dict(color='rgba(168,85,247,0.4)', width=1), showlegend=False))
        fig.add_trace(go.Scatter(x=valid_bb[date_col], y=valid_bb['BB_Lower'], name='BB Lower', line=dict(color='rgba(168,85,247,0.4)', width=1), fill='tonexty', fillcolor='rgba(168,85,247,0.05)', showlegend=False))
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df[date_col], y=df['Volume'], name='Volume', marker_color='rgba(161,161,170,0.15)', yaxis='y2'))

    fig.update_layout(**PLOTLY_TEMPLATE['layout'])
    fig.update_layout(height=480, yaxis=dict(title="Price ($)", gridcolor='#27272a'),
        yaxis2=dict(title="Volume", overlaying='y', side='right', showgrid=False,
                    range=[0, df['Volume'].max() * 4] if 'Volume' in df.columns else None),
        hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # ── KPI Row ──
    if 'Close' in df.columns:
        latest = df['Close'].iloc[-1]
        change = df['Close'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0
        trend_color = "#10b981" if change >= 0 else "#ef4444"
        trend_icon = "↑" if change >= 0 else "↓"
        rsi_val = f"{df['RSI_14'].iloc[-1]:.1f}" if 'RSI_14' in df.columns and df['RSI_14'].iloc[-1] > 0 else 'N/A'
        rsi_num = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 50
        rsi_label = 'Overbought' if rsi_num > 70 else 'Oversold' if rsi_num < 30 else 'Neutral'
        rsi_color = '#ef4444' if rsi_num > 70 else '#10b981' if rsi_num < 30 else '#a1a1aa'
        macd_val = f"{df['MACD'].iloc[-1]:.4f}" if 'MACD' in df.columns else 'N/A'
        vol_val = f"{df['Volatility_30'].iloc[-1]:.4f}" if 'Volatility_30' in df.columns and df['Volatility_30'].iloc[-1] > 0 else 'N/A'

        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-top: 16px;">
            <div class="bento-card" style="padding: 12px;">
                <div style="color: #a1a1aa; font-size: 0.7rem;">CLOSE</div>
                <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Geist Mono'; color: #f4f4f5;">${latest:.2f}
                    <span style="font-size: 0.75rem; color: {trend_color};">{trend_icon}{abs(change):.2f}%</span></div>
            </div>
            <div class="bento-card" style="padding: 12px;">
                <div style="color: #a1a1aa; font-size: 0.7rem;">52W HIGH</div>
                <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Geist Mono'; color: #f4f4f5;">${df['Close'].max():.2f}</div>
            </div>
            <div class="bento-card" style="padding: 12px;">
                <div style="color: #a1a1aa; font-size: 0.7rem;">52W LOW</div>
                <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Geist Mono'; color: #f4f4f5;">${df['Close'].min():.2f}</div>
            </div>
            <div class="bento-card" style="padding: 12px;">
                <div style="color: #a1a1aa; font-size: 0.7rem;">RSI (14) — Momentum</div>
                <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Geist Mono'; color: #f4f4f5;">{rsi_val}</div>
                <div style="font-size: 0.6rem; color: {rsi_color};">{rsi_label}</div>
            </div>
            <div class="bento-card" style="padding: 12px;">
                <div style="color: #a1a1aa; font-size: 0.7rem;">MACD — Trend Signal</div>
                <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Geist Mono'; color: #f4f4f5;">{macd_val}</div>
                <div style="font-size: 0.6rem; color: #71717a;">{'Bullish' if macd_val != 'N/A' and float(macd_val) > 0 else 'Bearish'}</div>
            </div>
            <div class="bento-card" style="padding: 12px;">
                <div style="color: #a1a1aa; font-size: 0.7rem;">VOL (30D) — Risk</div>
                <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Geist Mono'; color: #f4f4f5;">{vol_val}</div>
                <div style="font-size: 0.6rem; color: #71717a;">30-day price volatility</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Real Model Inference ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom: 4px;">🤖 AI Model Predictions (Real Inference)</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#71717a;font-size:0.75rem;margin-bottom:16px;">The system loads the trained neural network weights, runs forward inference on the latest available features for this ticker, and outputs dollar-denominated price targets. Confidence reflects how strongly the 4 models agree on the direction and magnitude.</div>', unsafe_allow_html=True)

    with st.spinner(f"Running ensemble inference for {selected_ticker}..."):
        preds = run_prediction_for_ticker(selected_ticker, n_samples=5)

    if preds:
        latest_pred = preds[-1]
        conf = latest_pred['confidence']
        conf_color = '#10b981' if conf > 0.7 else '#f59e0b' if conf > 0.5 else '#ef4444'
        last_close = df['Close'].iloc[-1] if 'Close' in df.columns else 0

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            v1 = latest_pred["pred_1d"]
            d1 = ((v1 - last_close) / last_close * 100) if last_close else 0
            c1_color = '#10b981' if d1 >= 0 else '#ef4444'
            st.markdown(f'<div class="bento-card" style="padding:16px;text-align:center;border-top:3px solid #3b82f6;"><div style="color:#a1a1aa;font-size:0.7rem;">1-DAY FORECAST</div><div style="font-size:1.4rem;font-weight:700;font-family:\'Geist Mono\';color:#3b82f6;">${v1:.2f}</div><div style="font-size:0.75rem;color:{c1_color};">{"↑" if d1>=0 else "↓"}{abs(d1):.2f}%</div></div>', unsafe_allow_html=True)
        with p2:
            v5 = latest_pred["pred_5d"]
            d5 = ((v5 - last_close) / last_close * 100) if last_close else 0
            c5_color = '#10b981' if d5 >= 0 else '#ef4444'
            st.markdown(f'<div class="bento-card" style="padding:16px;text-align:center;border-top:3px solid #a855f7;"><div style="color:#a1a1aa;font-size:0.7rem;">5-DAY FORECAST</div><div style="font-size:1.4rem;font-weight:700;font-family:\'Geist Mono\';color:#a855f7;">${v5:.2f}</div><div style="font-size:0.75rem;color:{c5_color};">{"↑" if d5>=0 else "↓"}{abs(d5):.2f}%</div></div>', unsafe_allow_html=True)
        with p3:
            v30 = latest_pred["pred_30d"]
            d30 = ((v30 - last_close) / last_close * 100) if last_close else 0
            c30_color = '#10b981' if d30 >= 0 else '#ef4444'
            st.markdown(f'<div class="bento-card" style="padding:16px;text-align:center;border-top:3px solid #f59e0b;"><div style="color:#a1a1aa;font-size:0.7rem;">30-DAY FORECAST</div><div style="font-size:1.4rem;font-weight:700;font-family:\'Geist Mono\';color:#f59e0b;">${v30:.2f}</div><div style="font-size:0.75rem;color:{c30_color};">{"↑" if d30>=0 else "↓"}{abs(d30):.2f}%</div></div>', unsafe_allow_html=True)
        with p4:
            st.markdown(f'<div class="bento-card" style="padding:16px;text-align:center;border-top:3px solid {conf_color};"><div style="color:#a1a1aa;font-size:0.7rem;">CONFIDENCE</div><div style="font-size:1.4rem;font-weight:700;font-family:\'Geist Mono\';color:{conf_color};">{conf:.1%}</div><div style="font-size:0.75rem;color:#71717a;">Ensemble agreement</div></div>', unsafe_allow_html=True)

        # ── Forecast Trajectory Chart ──
        st.markdown('<div class="card-title" style="margin-top:24px;margin-bottom:12px;">📈 Price Forecast Trajectory</div>', unsafe_allow_html=True)
        import datetime
        last_date = pd.to_datetime(df[date_col].iloc[-1])
        trajectory_dates = [last_date, last_date + datetime.timedelta(days=1), last_date + datetime.timedelta(days=5), last_date + datetime.timedelta(days=30)]
        trajectory_vals = [last_close, v1, v5, v30]
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(x=df[date_col].iloc[-60:], y=df['Close'].iloc[-60:], name='Historical', line=dict(color='#f4f4f5', width=1.5)))
        fig_traj.add_trace(go.Scatter(x=trajectory_dates, y=trajectory_vals, name='Forecast', mode='lines+markers',
            line=dict(color='#3b82f6', width=2, dash='dash'), marker=dict(size=8, color=['#f4f4f5','#3b82f6','#a855f7','#f59e0b'])))
        fig_traj.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig_traj.update_layout(height=300, yaxis_title="Price ($)", hovermode='x unified')
        st.plotly_chart(fig_traj, use_container_width=True)

        # Ensemble breakdown
        ew = latest_pred.get('ensemble_weights', {})
        ts_ens = latest_pred.get('ts_ensemble', {})

        ec1, ec2 = st.columns(2)
        with ec1:
            if ew:
                st.markdown('<div class="card-title" style="margin-top:16px;margin-bottom:12px;">Ensemble Weights (Regime-Adaptive)</div>', unsafe_allow_html=True)
                ew_fig = go.Figure(go.Bar(x=list(ew.keys()), y=list(ew.values()), marker_color=['#3b82f6','#10b981','#f59e0b','#a855f7'][:len(ew)],
                    text=[f"{v:.0%}" for v in ew.values()], textposition='auto'))
                ew_fig.update_layout(**PLOTLY_TEMPLATE['layout'])
                ew_fig.update_layout(height=250, yaxis_title="Weight")
                st.plotly_chart(ew_fig, use_container_width=True)
        with ec2:
            if ts_ens:
                st.markdown('<div class="card-title" style="margin-top:16px;margin-bottom:12px;">Individual Model Outputs</div>', unsafe_allow_html=True)
                ts_fig = go.Figure(go.Bar(x=list(ts_ens.keys()), y=list(ts_ens.values()), marker_color=['#3b82f6','#10b981','#f59e0b','#a855f7'][:len(ts_ens)],
                    text=[f"{v:.4f}" for v in ts_ens.values()], textposition='auto'))
                ts_fig.update_layout(**PLOTLY_TEMPLATE['layout'])
                ts_fig.update_layout(height=250, yaxis_title="Raw Prediction")
                st.plotly_chart(ts_fig, use_container_width=True)

        # ── Historical Predictions Table ──
        if len(preds) > 1:
            st.markdown('<div class="card-title" style="margin-top:24px;margin-bottom:12px;">📋 Recent Prediction History</div>', unsafe_allow_html=True)
            pred_df = pd.DataFrame([{
                'Date': p['date'], '1-Day ($)': f"${p['pred_1d']:.2f}", '5-Day ($)': f"${p['pred_5d']:.2f}",
                '30-Day ($)': f"${p['pred_30d']:.2f}", 'Confidence': f"{p['confidence']:.1%}",
            } for p in preds])
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
    else:
        # Check if ticker exists in test metadata
        meta = load_test_metadata()
        if meta is not None:
            available = meta['ticker'].nunique()
            has_ticker = selected_ticker in meta['ticker'].values if meta is not None else False
            if has_ticker:
                st.warning(f"⚠️ Model loading issue for {selected_ticker}. The test data exists but model inference failed. Check the Streamlit console for errors.")
            else:
                st.info(f"📊 **{selected_ticker}** is not in the test dataset ({available} tickers available). Try selecting a ticker from the test set (e.g., AAPL, MSFT, GOOGL).")
        else:
            st.info(f"Model inference not available. Run the training pipeline first to generate test data (`X_test.npy`/`metadata_test.csv`).")

    # ── Market Context Panel ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:4px;">🌐 Market Context — Multi-Domain Signal Feed</div>', unsafe_allow_html=True)
    st.markdown('''
    <div style="color:#71717a;font-size:0.75rem;margin-bottom:16px;">
        These real-time signals from macro-economic, energy, and social domains are injected as contextual features into the prediction models.
        <strong style="color:#a1a1aa;">Market Regime</strong> determines ensemble weights |
        <strong style="color:#a1a1aa;">VIX</strong> measures market fear (>25 = elevated risk) |
        <strong style="color:#a1a1aa;">Crude Oil</strong> proxies global supply chain health |
        <strong style="color:#a1a1aa;">Social Signals</strong> track developer/community activity.
    </div>
    ''', unsafe_allow_html=True)

    ctx1, ctx2, ctx3, ctx4 = st.columns(4)

    # Regime context
    with ctx1:
        regime_df = load_regime_states()
        if regime_df is not None and not regime_df.empty:
            current_regime = int(regime_df['regime'].iloc[-1])
            r_label, r_color = get_regime_label(current_regime)
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;border-top:3px solid {r_color};"><div style="color:#a1a1aa;font-size:0.6rem;">MARKET REGIME</div><div style="font-size:0.9rem;font-weight:700;color:{r_color};">{r_label}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.6rem;">MARKET REGIME</div><div style="color:#71717a;">N/A</div></div>', unsafe_allow_html=True)

    # FRED macro context
    with ctx2:
        fred_df = load_economic_indicators()
        if fred_df is not None and 'indicator' in fred_df.columns:
            vix = fred_df[fred_df['indicator'] == 'VIX_INDEX']
            vix_val = f"{vix['value'].iloc[-1]:.1f}" if not vix.empty else 'N/A'
            vix_color = '#ef4444' if not vix.empty and vix['value'].iloc[-1] > 25 else '#10b981'
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;border-top:3px solid {vix_color};"><div style="color:#a1a1aa;font-size:0.6rem;">VIX (FEAR INDEX)</div><div style="font-size:0.9rem;font-weight:700;color:{vix_color};">{vix_val}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.6rem;">VIX</div><div style="color:#71717a;">N/A</div></div>', unsafe_allow_html=True)

    # Energy context
    with ctx3:
        energy_df = load_energy_data()
        if energy_df is not None and 'series' in energy_df.columns:
            oil = energy_df[energy_df['series'] == 'crude_oil_production']
            oil_val = f"{oil['value'].iloc[-1]:,.0f}" if not oil.empty else 'N/A'
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;border-top:3px solid #f59e0b;"><div style="color:#a1a1aa;font-size:0.6rem;">CRUDE OIL PROD</div><div style="font-size:0.9rem;font-weight:700;color:#f59e0b;">{oil_val}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.6rem;">ENERGY</div><div style="color:#71717a;">N/A</div></div>', unsafe_allow_html=True)

    # Social sentiment context
    with ctx4:
        social_df = load_social_signals()
        if social_df is not None and not social_df.empty:
            # Get latest overall sentiment proxy by isolating explicit sentiment columns
            sent_cols = [c for c in social_df.columns if 'sentiment' in c.lower()]
            # Also check for engagement metrics as social signal strength
            engagement_cols = [c for c in social_df.columns if 'engagement' in c.lower() or 'stars' in c.lower()]
            if sent_cols:
                # Average the latest sentiment values across platforms, scale to -100 to +100 for readability
                raw_sent = social_df[sent_cols].iloc[-1].mean()
                sent_val = raw_sent * 100
                sent_color = '#10b981' if sent_val > 0 else '#ef4444' if sent_val < 0 else '#a1a1aa'
                # Provide qualitative label alongside numerical score
                sent_label = "Bullish" if sent_val > 2.5 else "Bearish" if sent_val < -2.5 else "Neutral"
                st.markdown(f'''
                <div class="bento-card" style="padding:10px;text-align:center;border-top:3px solid {sent_color};">
                    <div style="color:#a1a1aa;font-size:0.6rem;text-transform:uppercase;">Social Sentiment</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{sent_color};font-family:'Geist Mono';">{sent_val:+.1f}</div>
                    <div style="font-size:0.6rem;color:{sent_color};margin-top:2px;">{sent_label}</div>
                </div>
                ''', unsafe_allow_html=True)
            elif engagement_cols:
                eng_val = social_df[engagement_cols[0]].iloc[-1]
                st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;border-top:3px solid #3b82f6;"><div style="color:#a1a1aa;font-size:0.6rem;">SOCIAL ACTIVITY</div><div style="font-size:0.9rem;font-weight:700;color:#3b82f6;">{eng_val:,.0f}</div></div>', unsafe_allow_html=True)
            else:
                days = len(social_df)
                st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;border-top:3px solid #3b82f6;"><div style="color:#a1a1aa;font-size:0.6rem;">SOCIAL SIGNALS</div><div style="font-size:0.9rem;font-weight:700;color:#3b82f6;">{days:,} days</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.6rem;">SOCIAL</div><div style="color:#71717a;">N/A</div></div>', unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def page_anomaly_detection():
    render_section_header("Anomaly & Regime Detection", "HMM market regime states + statistical anomalies from real data")

    st.markdown('''
    <div style="background: rgba(239,68,68,0.05); border: 1px solid #27272a; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
        <div style="color:#a1a1aa; font-size: 0.78rem; line-height: 1.5;">
            🎯 <strong style="color:#f4f4f5;">Market Regimes:</strong> A <strong style="color:#3b82f6;">5-state Hidden Markov Model (HMM)</strong> trained on historical returns
            automatically classifies market conditions as Bull, Bear, Sideways, High Volatility, or Trending.
            The detected regime <strong>dynamically adjusts ensemble model weights</strong> — e.g., in High Volatility, conservative models
            get higher weight. <strong style="color:#ef4444;">Anomalies</strong> are detected using Z-score analysis (|Z| > 2.5 = statistically extreme move).
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # ── Real Regime States ──
    regime_df = load_regime_states()
    if regime_df is not None and not regime_df.empty:
        st.markdown('<div class="card-title" style="margin-bottom: 16px;">🔮 Market Regime States (Trained HMM — 5 States)</div>', unsafe_allow_html=True)
        current_regime = int(regime_df['regime'].iloc[-1])
        label, color = get_regime_label(current_regime)
        st.markdown(f'<div class="bento-card" style="padding:16px; border-left: 4px solid {color};"><span style="font-size:0.8rem;color:#a1a1aa;">Current Market Regime</span><br/><span style="font-size:1.3rem;font-weight:700;color:{color};">{label}</span><span style="color:#a1a1aa;font-size:0.85rem;margin-left:12px;">as of {regime_df["date"].iloc[-1].strftime("%Y-%m-%d")}</span></div>', unsafe_allow_html=True)

        fig_regime = go.Figure()
        for rid, (rl, rc) in REGIME_LABELS.items():
            mask = regime_df['regime'] == rid
            if mask.any():
                fig_regime.add_trace(go.Scatter(x=regime_df.loc[mask, 'date'], y=regime_df.loc[mask, 'regime'], mode='markers', name=rl, marker=dict(color=rc, size=4)))
        fig_regime.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig_regime.update_layout(height=250, yaxis_title="Regime ID", yaxis=dict(dtick=1), showlegend=True)
        st.plotly_chart(fig_regime, use_container_width=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            dist = regime_df['regime'].value_counts().sort_index()
            labels_list = [get_regime_label(i)[0] for i in dist.index]
            colors_list = [get_regime_label(i)[1] for i in dist.index]
            fig_pie = go.Figure(go.Pie(labels=labels_list, values=dist.values, hole=0.5, marker=dict(colors=colors_list), textinfo='label+percent'))
            fig_pie.update_layout(**PLOTLY_TEMPLATE['layout'])
            fig_pie.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        with rc2:
            fig_bar = go.Figure(go.Bar(x=labels_list, y=dist.values, marker_color=colors_list, text=dist.values, textposition='auto'))
            fig_bar.update_layout(**PLOTLY_TEMPLATE['layout'])
            fig_bar.update_layout(height=300)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No regime states found. Run `scripts/generate_global_regime.py`.")

    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom: 16px;">📊 Statistical Anomaly Detection</div>', unsafe_allow_html=True)

    tickers = list_available_tickers()
    if not tickers:
        return

    selected = st.selectbox("Select Asset for Analysis", tickers[:30])
    df = load_stock_data(selected)
    if df is None or 'Close' not in df.columns:
        st.warning("No price data available.")
        return

    date_col = 'Date' if 'Date' in df.columns else 'date'
    df['Returns'] = df['Close'].pct_change()
    df['Z_Score'] = (df['Returns'] - df['Returns'].mean()) / df['Returns'].std()
    df['Anomaly'] = df['Z_Score'].abs() > 2.5
    anomalies = df[df['Anomaly']].copy()

    st.markdown(f'<div style="color:#a1a1aa;margin-bottom:12px;">🚨 {len(anomalies)} anomalous events in {selected}</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df['Close'], name='Price', line=dict(color='#3b82f6', width=1.5)))
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies[date_col], y=anomalies['Close'], name='Anomaly', mode='markers', marker=dict(color='#ef4444', size=8)))
    fig.update_layout(**PLOTLY_TEMPLATE['layout'])
    fig.update_layout(height=380, yaxis_title="Price", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_h = go.Figure(go.Histogram(x=df['Returns'].dropna(), nbinsx=50, marker_color='#10b981'))
        fig_h.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig_h.update_layout(height=280, xaxis_title="Daily Return")
        st.plotly_chart(fig_h, use_container_width=True)
    with c2:
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=df[date_col], y=df['Z_Score'], line=dict(color='#10b981', width=1)))
        fig_z.add_hline(y=2.5, line_dash="dash", line_color='#ef4444')
        fig_z.add_hline(y=-2.5, line_dash="dash", line_color='#ef4444')
        fig_z.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig_z.update_layout(height=280, yaxis_title="Z-Score")
        st.plotly_chart(fig_z, use_container_width=True)

    if not anomalies.empty:
        display_df = anomalies[[date_col, 'Close', 'Returns', 'Z_Score']].copy()
        display_df['Returns'] = (display_df['Returns'] * 100).round(2)
        display_df['Z_Score'] = display_df['Z_Score'].round(2)
        display_df.columns = ['Date', 'Price', 'Return (%)', 'Z-Score']
        st.dataframe(display_df.sort_values('Date', ascending=False), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZATION WORKSPACE
# ═══════════════════════════════════════════════════════════════════════════════

def page_visualization():
    render_section_header("Visualization Workspace", "Explore real stock, crypto, and feature data interactively")

    st.markdown('<div style="color:#71717a;font-size:0.75rem;margin-bottom:12px;">Select a mode below to interactively explore any dataset in the platform. Use this workspace to discover patterns, correlations, and anomalies across all collected data domains.</div>', unsafe_allow_html=True)

    viz_mode = st.selectbox("Visualization Mode", ["Stock Explorer", "Multi-Ticker Comparison", "Technical Indicator Heatmap", "Crypto Explorer", "Economic Explorer", "Energy Explorer", "Weather Explorer"])

    if viz_mode == "Stock Explorer":
        tickers = list_available_tickers()
        if not tickers:
            st.info("No stock data available.")
            return
        sel = st.selectbox("Ticker", tickers, index=tickers.index('AAPL') if 'AAPL' in tickers else 0)
        df = load_stock_data(sel)
        if df is None or df.empty:
            st.warning("No data.")
            return
        date_col = 'Date' if 'Date' in df.columns else 'date'
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1:
            chart_type = st.selectbox("Chart Type", ["Line", "Scatter", "Histogram", "Correlation Heatmap"])
        with c2:
            y_col = st.selectbox("Y Axis", numeric_cols, index=numeric_cols.index('Close') if 'Close' in numeric_cols else 0)
        with c3:
            y2_col = st.selectbox("Overlay (optional)", ['None'] + numeric_cols)

        st.markdown('<div style="height: 1px; background: #27272a; margin: 24px 0;"></div>', unsafe_allow_html=True)
        if chart_type == "Correlation Heatmap":
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, color_continuous_scale=['#09090b', '#3b82f6', '#f4f4f5'], aspect='auto', text_auto='.2f')
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=y_col, nbins=50)
            fig.update_traces(marker_color='#10b981')
        elif chart_type == "Scatter":
            x_col_s = st.selectbox("X Axis", numeric_cols, index=numeric_cols.index('Volume') if 'Volume' in numeric_cols else 1)
            fig = px.scatter(df, x=x_col_s, y=y_col, opacity=0.6)
            fig.update_traces(marker=dict(color='#3b82f6', size=4))
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[date_col], y=df[y_col], name=y_col, line=dict(color='#3b82f6', width=1.5)))
            if y2_col != 'None':
                fig.add_trace(go.Scatter(x=df[date_col], y=df[y2_col], name=y2_col, yaxis='y2', line=dict(color='#f59e0b', width=1.5)))
                fig.update_layout(yaxis2=dict(title=y2_col, overlaying='y', side='right', showgrid=False))
        fig.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="card-title" style="margin-top:24px;margin-bottom:12px;">Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.tail(20), use_container_width=True, hide_index=True)

    elif viz_mode == "Multi-Ticker Comparison":
        tickers = list_available_tickers()
        selected = st.multiselect("Compare Tickers", tickers, default=[t for t in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'] if t in tickers][:4])
        metric = st.selectbox("Metric", ['Close', 'Volume', 'Daily_Return', 'RSI_14', 'MACD', 'Volatility_30'])
        if selected:
            fig = go.Figure()
            for t in selected:
                df = load_stock_data(t)
                if df is not None and metric in df.columns:
                    dc = 'Date' if 'Date' in df.columns else 'date'
                    fig.add_trace(go.Scatter(x=df[dc], y=df[metric], name=t, line=dict(width=1.5)))
            fig.update_layout(**PLOTLY_TEMPLATE['layout'])
            fig.update_layout(height=500, yaxis_title=metric, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

    elif viz_mode == "Technical Indicator Heatmap":
        tickers = list_available_tickers()
        sel = st.selectbox("Ticker", tickers, index=tickers.index('AAPL') if 'AAPL' in tickers else 0, key='ti_ticker')
        df = load_stock_data(sel)
        if df is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Filter to valid data (non-zero)
            valid = df[numeric_cols].iloc[60:]  # Skip warmup period
            corr = valid.corr()
            fig = px.imshow(corr, color_continuous_scale=['#09090b', '#3b82f6', '#f4f4f5'], aspect='auto', text_auto='.2f')
            fig.update_layout(**PLOTLY_TEMPLATE['layout'])
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

    elif viz_mode == "Crypto Explorer":
        crypto_df = load_crypto_data()
        if crypto_df is not None and not crypto_df.empty:
            coins = sorted(crypto_df['coin'].unique()) if 'coin' in crypto_df.columns else []
            sel_coin = st.selectbox("Coin", coins, index=coins.index('bitcoin') if 'bitcoin' in coins else 0)
            coin_data = crypto_df[crypto_df['coin'] == sel_coin].sort_values('date' if 'date' in crypto_df.columns else 'timestamp')
            dc = 'date' if 'date' in coin_data.columns else 'timestamp'
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=coin_data[dc], y=coin_data['price'], name=sel_coin.title(), line=dict(color='#f59e0b', width=2)))
            fig.update_layout(**PLOTLY_TEMPLATE['layout'])
            fig.update_layout(height=450, yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(coin_data.tail(20), use_container_width=True, hide_index=True)
        else:
            st.info("No crypto data available.")

    elif viz_mode == "Economic Explorer":
        fred_df = load_economic_indicators()
        if fred_df is not None and not fred_df.empty and 'indicator' in fred_df.columns:
            indicators = sorted(fred_df['indicator'].unique())
            selected = st.multiselect("Select FRED Indicators", indicators,
                default=[i for i in ['GDP', 'CPI', 'UNEMPLOYMENT_RATE', 'VIX_INDEX'] if i in indicators][:3])
            if selected:
                fig = go.Figure()
                for ind in selected:
                    data = fred_df[fred_df['indicator'] == ind].sort_values('date')
                    fig.add_trace(go.Scatter(x=data['date'], y=data['value'], name=ind.replace('_', ' ').title(), line=dict(width=1.5)))
                fig.update_layout(**PLOTLY_TEMPLATE['layout'])
                fig.update_layout(height=500, yaxis_title="Value", hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                # Show data summary
                for ind in selected:
                    data = fred_df[fred_df['indicator'] == ind]
                    if not data.empty:
                        st.markdown(f"**{ind.replace('_', ' ').title()}**: {len(data):,} data points, range {data['date'].min().strftime('%Y')}–{data['date'].max().strftime('%Y')}, latest = {data['value'].iloc[-1]:,.2f}")
        else:
            st.info("No economic data available.")

    elif viz_mode == "Energy Explorer":
        energy_df = load_energy_data()
        if energy_df is not None and not energy_df.empty and 'series' in energy_df.columns:
            series_list = sorted(energy_df['series'].unique())
            sel_series = st.multiselect("Select Energy Series", series_list, default=series_list[:3])
            if sel_series:
                fig = go.Figure()
                for s in sel_series:
                    data = energy_df[energy_df['series'] == s].sort_values('date')
                    fig.add_trace(go.Scatter(x=data['date'], y=data['value'], name=s.replace('_', ' ').title(), line=dict(width=1.5)))
                fig.update_layout(**PLOTLY_TEMPLATE['layout'])
                fig.update_layout(height=500, yaxis_title="Production/Demand", hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(energy_df[energy_df['series'].isin(sel_series)].tail(30), use_container_width=True, hide_index=True)
        else:
            st.info("No energy data available.")

    elif viz_mode == "Weather Explorer":
        weather_df = load_weather_data()
        if weather_df is not None and not weather_df.empty:
            metric_w = st.selectbox("Weather Metric", ['temperature', 'humidity', 'pressure', 'wind_speed'])
            if metric_w in weather_df.columns and 'city' in weather_df.columns:
                sorted_df = weather_df.sort_values(metric_w, ascending=False)
                fig = go.Figure(go.Bar(x=sorted_df['city'], y=sorted_df[metric_w],
                    marker_color=[f'hsl({int(i * 360 / len(sorted_df))}, 70%, 55%)' for i in range(len(sorted_df))],
                    text=[f"{v:.1f}" for v in sorted_df[metric_w]], textposition='auto'))
                fig.update_layout(**PLOTLY_TEMPLATE['layout'])
                fig.update_layout(height=500, yaxis_title=metric_w.replace('_', ' ').title())
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(weather_df.sort_values('city'), use_container_width=True, hide_index=True)
        else:
            st.info("No weather data available.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SYSTEM MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

def page_system_monitor():
    render_section_header("System Monitor", "Infrastructure, tech stack, and resource usage")

    stats = get_system_stats()

    def make_gauge(value, title, color):
        fig = go.Figure(go.Indicator(mode="gauge+number", value=value,
            title=dict(text=title, font=dict(size=14, color='#a1a1aa')),
            number=dict(suffix="%", font=dict(color=color, size=32, family='Geist Mono')),
            gauge=dict(axis=dict(range=[0, 100], tickcolor='#27272a', dtick=25, tickfont=dict(color='#a1a1aa')),
                bar=dict(color=color), bgcolor='#09090b', borderwidth=1, bordercolor='#27272a',
                steps=[dict(range=[0, 50], color='rgba(59,130,246,0.05)'), dict(range=[50, 80], color='rgba(245,158,11,0.05)'), dict(range=[80, 100], color='rgba(239,68,68,0.05)')],
                threshold=dict(line=dict(color='#ef4444', width=2), thickness=0.8, value=85))))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=240, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    cpu_color = '#ef4444' if stats['cpu_percent'] > 80 else '#f59e0b' if stats['cpu_percent'] > 50 else '#3b82f6'
    mem_color = '#ef4444' if stats['memory_percent'] > 80 else '#f59e0b' if stats['memory_percent'] > 50 else '#3b82f6'
    disk_color = '#ef4444' if stats['disk_percent'] > 80 else '#f59e0b' if stats['disk_percent'] > 50 else '#3b82f6'

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(make_gauge(stats['cpu_percent'], 'CPU Usage', cpu_color), use_container_width=True)
    with g2:
        st.plotly_chart(make_gauge(stats['memory_percent'], 'Memory Usage', mem_color), use_container_width=True)
    with g3:
        st.plotly_chart(make_gauge(stats['disk_percent'], 'Disk Usage', disk_color), use_container_width=True)

    s1, s2, s3 = st.columns(3)
    s1.metric("RAM Used", f"{stats['memory_used_gb']} GB", f"/ {stats['memory_total_gb']} GB")
    s2.metric("Disk Used", f"{stats['disk_used_gb']} GB", f"/ {stats['disk_total_gb']} GB")
    s3.metric("CPU Cores", str(os.cpu_count()))

    # ── Technology Stack ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:16px;">🛠 Technology Stack Used</div>', unsafe_allow_html=True)

    tech_stack = {
        "Deep Learning": [
            ("LSTM", "Long Short-Term Memory — sequential time series forecasting"),
            ("GRU", "Gated Recurrent Unit — efficient sequential modeling"),
            ("Transformer", "Self-attention encoder for market pattern recognition"),
            ("TFT", "Temporal Fusion Transformer — multi-horizon probabilistic forecasting"),
            ("DeBERTa-v3", "NLP multi-task model for sentiment, event, topic classification"),
        ],
        "Ensemble & Fusion": [
            ("Cross-Attention Fusion", "Combines TS + NLP embeddings via learned attention weights"),
            ("Regime-Adaptive Ensemble", "HMM-driven dynamic weight allocation across models"),
            ("Multi-Horizon Output", "Simultaneous 1-day, 5-day, 30-day return predictions"),
        ],
        "Data Pipeline": [
            ("yfinance", "S&P 500 stock data — 502 tickers, 10+ years of OHLCV"),
            ("CoinGecko", "Top 100 cryptocurrency prices and market cap data"),
            ("FRED/BLS", "Macroeconomic indicators — GDP, CPI, unemployment, yields"),
            ("Social Scraper", "GitHub, HackerNews, YouTube, Mastodon, StackExchange, Google Trends"),
            ("OECD/World Bank", "Alternative data — aviation, blockchain, patents, trade, jobs"),
        ],
        "Infrastructure": [
            ("PyTorch + MPS", "Apple Silicon GPU acceleration for model training"),
            ("HuggingFace Transformers", "DeBERTa-v3 base for NLP multi-task learning"),
            ("hmmlearn", "Hidden Markov Model for 5-state market regime detection"),
            ("Streamlit", "Real-time interactive dashboard with live data binding"),
            ("FastAPI", "REST API backend for model inference serving"),
            ("Parquet + NumPy", "Efficient data storage for 1.5M+ datapoints"),
        ],
    }

    for section, items in tech_stack.items():
        st.markdown(f'<div class="card-title" style="margin-top:24px;margin-bottom:8px;font-size:0.9rem;">{section}</div>', unsafe_allow_html=True)
        cards_html = '<div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(280px, 1fr));gap:8px;">'
        for name, desc in items:
            cards_html += f'<div class="bento-card" style="padding:10px;"><strong style="color:#3b82f6;font-size:0.85rem;">{name}</strong><div style="color:#a1a1aa;font-size:0.7rem;margin-top:4px;">{desc}</div></div>'
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── Top Processes ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:16px;">Top Processes (by CPU)</div>', unsafe_allow_html=True)
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            info = p.info
            if info.get('cpu_percent', 0) and info['cpu_percent'] > 0.1:
                procs.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    procs.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
    if procs:
        df_procs = pd.DataFrame(procs[:15])
        df_procs.columns = ['PID', 'Process', 'CPU %', 'Memory %']
        st.dataframe(df_procs, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LOGS
# ═══════════════════════════════════════════════════════════════════════════════

def page_logs():
    render_section_header("System Logs", "Recent pipeline and training logs")
    logs = get_recent_logs(100)
    log_text = ''.join(logs)
    st.code(log_text if log_text.strip() else "No log entries found.", language="log")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

def page_settings():
    render_section_header("Settings", "Platform configuration and technology descriptions")

    # Technology descriptions for config keys
    key_descriptions = {
        'ALPHA_VANTAGE_API_KEY': ('📊 Alpha Vantage', 'Financial market data — earnings, technical indicators, sector performance'),
        'ALPHA_VANTAGE_KEY': ('📊 Alpha Vantage', 'Financial market data — earnings, technical indicators, sector performance'),
        'FRED_API_KEY': ('🏛 FRED', 'Federal Reserve Economic Data — GDP, CPI, unemployment, interest rates, yield curves'),
        'FRED': ('🏛 FRED', 'Federal Reserve Economic Data — GDP, CPI, unemployment, interest rates, yield curves'),
        'NEWS_API_KEY': ('📰 NewsAPI', 'News article aggregation for NLP sentiment analysis across global sources'),
        'NEWSAPI_KEY': ('📰 NewsAPI', 'News article aggregation for NLP sentiment analysis across global sources'),
        'POLYGON_API_KEY': ('📈 Polygon.io', 'Real-time and historical market data, options, and forex'),
        'FINNHUB_API_KEY': ('🔍 Finnhub', 'Real-time stock prices, company financials, and economic calendar'),
        'QUANDL_API_KEY': ('📉 Quandl/Nasdaq', 'Alternative financial datasets — commodities, futures, economic data'),
        'OPENAI_API_KEY': ('🤖 OpenAI', 'LLM-powered text analysis and embedding generation'),
        'BLS_API_KEY': ('👷 BLS', 'Bureau of Labor Statistics — employment data and job market indicators'),
        'WORLD_BANK_API_KEY': ('🌍 World Bank', 'Global development indicators and international trade data'),
        'YOUTUBE_CLIENT_ID': ('🎥 YouTube', 'YouTube Data API — tech video engagement, views, and content tracking'),
        'YOUTUBE_CLIENT_SECRET': ('🎥 YouTube Secret', 'YouTube OAuth client secret for API authentication'),
        'YOUTUBE_API_KEY': ('🎥 YouTube API', 'YouTube Data API key for public data queries'),
        'SERPAPI_KEY': ('🔎 SerpAPI', 'Google Trends & search data — real-time search interest for market sentiment'),
        'MASTODON_ACCESS_TOKEN': ('🐘 Mastodon', 'Mastodon social network — federated social signals for NLP analysis'),
        'GITHUB_TOKEN': ('🐙 GitHub', 'GitHub API — repository metrics, stars, forks, and developer activity'),
    }

    st.markdown('<div class="card-title" style="margin-bottom:16px;">🔧 Data Source Configuration</div>', unsafe_allow_html=True)
    env_path = os.path.join(get_project_root(), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    val = val.strip().strip('"').strip("'")
                    configured = bool(val and len(val) > 4)
                    badge_type = 'badge-active' if configured else 'badge-neutral'
                    badge_dot = "<div class='badge-dot'></div>"
                    status_text = 'Configured' if configured else 'Not Set'

                    desc_info = key_descriptions.get(key.strip(), (f'🔑 {key.strip()}', 'Custom configuration key'))
                    icon_name, description = desc_info

                    st.markdown(f"""
                    <div class="bento-card" style="padding: 14px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="color: #f4f4f5; font-weight: 600; font-size: 0.9rem;">{icon_name}</span>
                                <span class="badge {badge_type}" style="margin-left: 12px;">{badge_dot}{status_text}</span>
                            </div>
                        </div>
                        <div style="color: #a1a1aa; font-size: 0.75rem; margin-top: 6px;">{description}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("No .env file found.")

    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)

    # Pipeline configuration
    st.markdown('<div class="card-title" style="margin-bottom:16px;">⚙️ Pipeline Configuration</div>', unsafe_allow_html=True)
    config_path = os.path.join(get_project_root(), 'configs', 'training_config.yaml')
    if os.path.exists(config_path):
        import yaml
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            if config:
                for section, values in config.items():
                    items_html = ''
                    if isinstance(values, dict):
                        for k, v in values.items():
                            items_html += f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1a1a1a;"><span style="color:#a1a1aa;font-size:0.75rem;">{k}</span><span style="color:#f4f4f5;font-family:\'Geist Mono\';font-size:0.75rem;">{v}</span></div>'
                    else:
                        items_html = f'<span style="color:#f4f4f5;font-family:\'Geist Mono\';font-size:0.8rem;">{values}</span>'
                    st.markdown(f'<div class="bento-card" style="padding:12px;margin-bottom:8px;"><div style="color:#3b82f6;font-weight:600;font-size:0.85rem;margin-bottom:8px;">{section}</div>{items_html}</div>', unsafe_allow_html=True)
        except Exception:
            st.info("Could not parse training config.")

    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:16px;">📋 Project Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="bento-card">
        <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 12px;">
            <div style="color: #a1a1aa; font-size: 0.9rem;">Project Root</div>
            <div style="color: #f4f4f5; font-family: 'Geist Mono'; font-size: 0.9rem;">{get_project_root()}</div>
            <div style="color: #a1a1aa; font-size: 0.9rem;">Python Version</div>
            <div style="color: #f4f4f5; font-family: 'Geist Mono'; font-size: 0.9rem;">{sys.version.split()[0]}</div>
            <div style="color: #a1a1aa; font-size: 0.9rem;">Streamlit Version</div>
            <div style="color: #f4f4f5; font-family: 'Geist Mono'; font-size: 0.9rem;">{st.__version__}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SOCIAL INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

def page_social_intelligence():
    render_section_header("Social & NLP Intelligence", "Real collected social media signals and NLP analysis")

    st.markdown('''
    <div style="background: linear-gradient(135deg, rgba(168,85,247,0.06), rgba(6,182,212,0.05)); border: 1px solid #27272a; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
        <div style="color:#a1a1aa; font-size: 0.78rem; line-height: 1.5;">
            🧠 <strong style="color:#f4f4f5;">Social & NLP Intelligence Pipeline:</strong>
            Social signals from 6 platforms (GitHub, HackerNews, YouTube, Mastodon, StackExchange, Google Trends) are collected daily
            and aggregated into engagement metrics. A <strong style="color:#a855f7;">DeBERTa-v3 multi-task NLP model</strong> then classifies text data for
            <strong style="color:#3b82f6;">sentiment</strong> (positive/negative/neutral),
            <strong style="color:#10b981;">event type</strong> (crash, regulation, tech launch, etc.),
            and <strong style="color:#f59e0b;">topic category</strong> (AI, EV, crypto, etc.) — all fed as features into the prediction ensemble.
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # ── Social Signals ──
    social_df = load_social_signals()
    if social_df is not None and not social_df.empty:
        st.markdown(f'<div class="card-title" style="margin-bottom:16px;">📱 Social Media Signals ({len(social_df):,} days of data)</div>', unsafe_allow_html=True)

        # Platform definitions with their columns
        platforms = {
            'GitHub': {'cols': ['github_repo_count', 'github_stars_total', 'github_forks_total'], 'color': '#f97316', 'icon': '🐙'},
            'HackerNews': {'cols': ['hn_post_count', 'hn_score_total'], 'color': '#10b981', 'icon': '📰'},
            'YouTube': {'cols': ['yt_video_count', 'yt_views_total', 'yt_engagement_total'], 'color': '#ef4444', 'icon': '📺'},
            'Mastodon': {'cols': ['mastodon_post_count', 'mastodon_engagement_total'], 'color': '#a855f7', 'icon': '🐘'},
            'StackExchange': {'cols': ['se_post_count', 'se_score_total'], 'color': '#3b82f6', 'icon': '💬'},
            'Google Trends': {'cols': ['gt_trend_count', 'gt_traffic_total'], 'color': '#f59e0b', 'icon': '📊'},
        }

        # KPI summary — show data points per platform
        kpi_html = '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:24px;">'
        for pname, pinfo in platforms.items():
            avail_cols = [c for c in pinfo['cols'] if c in social_df.columns]
            total_points = sum((social_df[c] != 0).sum() for c in avail_cols) if avail_cols else 0
            kpi_html += f'''<div class="bento-card" style="padding:10px;text-align:center;border-top:2px solid {pinfo['color']};">
                <div style="font-size:1.2rem;">{pinfo['icon']}</div>
                <div style="color:#a1a1aa;font-size:0.65rem;">{pname}</div>
                <div style="font-size:1rem;font-weight:600;color:#f4f4f5;">{total_points:,}</div>
                <div style="color:#71717a;font-size:0.6rem;">data points</div></div>'''
        kpi_html += '</div>'
        st.markdown(kpi_html, unsafe_allow_html=True)

        # Charts — only show platforms with actual data, 2 per row
        active_platforms = []
        for pname, pinfo in platforms.items():
            avail_cols = [c for c in pinfo['cols'] if c in social_df.columns]
            has_data = any((social_df[c] != 0).sum() > 5 for c in avail_cols)
            if avail_cols and has_data:
                active_platforms.append((pname, pinfo, avail_cols))

        for i in range(0, len(active_platforms), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(active_platforms):
                    pname, pinfo, avail_cols = active_platforms[i + j]
                    with col:
                        fig = go.Figure()
                        for c in avail_cols:
                            non_zero = social_df[social_df[c] != 0]
                            if not non_zero.empty:
                                fig.add_trace(go.Scatter(x=non_zero['date'], y=non_zero[c],
                                    name=c.split('_', 1)[1].replace('_', ' ').title(),
                                    line=dict(width=1.5)))
                        fig.update_layout(**PLOTLY_TEMPLATE['layout'])
                        fig.update_layout(height=280, title=f"{pinfo['icon']} {pname}", showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No social signals. Run data collection first.")

    # ── NLP Signals ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    nlp_df = load_nlp_signals()
    if nlp_df is not None and not nlp_df.empty:
        st.markdown(f'<div class="card-title" style="margin-bottom:16px;">🧠 NLP Sentiment & Event Analysis ({len(nlp_df)} records)</div>', unsafe_allow_html=True)
        sent_cols = [c for c in nlp_df.columns if c.startswith('sentiment_')]
        topic_cols = [c for c in nlp_df.columns if c.startswith('topic_')]
        event_cols = [c for c in nlp_df.columns if c.startswith('event_')]

        if sent_cols:
            c1, c2 = st.columns(2)
            with c1:
                latest = nlp_df.iloc[-1]
                vals = [latest.get(c, 0) for c in sent_cols]
                labels = [c.replace('sentiment_', '').title() for c in sent_cols]
                fig_s = go.Figure(go.Pie(labels=labels, values=vals,
                    marker=dict(colors=['#10b981', '#ef4444', '#a1a1aa'][:len(vals)]),
                    hole=0.4, textinfo='label+percent'))
                fig_s.update_layout(**PLOTLY_TEMPLATE['layout'])
                fig_s.update_layout(height=300, title="Sentiment Distribution")
                st.plotly_chart(fig_s, use_container_width=True)
            with c2:
                if event_cols:
                    e_vals = [latest.get(c, 0) for c in event_cols]
                    e_names = [c.replace('event_', '').title() for c in event_cols]
                    # Filter only non-zero events
                    active = [(n, v) for n, v in zip(e_names, e_vals) if v > 0]
                    if active:
                        fig_e = go.Figure(go.Bar(x=[a[0] for a in active], y=[a[1] for a in active],
                            marker_color='#3b82f6', text=[f"{v:.2f}" for _, v in active], textposition='auto'))
                    else:
                        fig_e = go.Figure(go.Bar(x=e_names, y=e_vals, marker_color='#3b82f6'))
                    fig_e.update_layout(**PLOTLY_TEMPLATE['layout'])
                    fig_e.update_layout(height=300, title="Event Detection Scores")
                    st.plotly_chart(fig_e, use_container_width=True)

        if topic_cols:
            latest = nlp_df.iloc[-1]
            t_vals = [latest.get(c, 0) for c in topic_cols]
            t_names = [c.replace('topic_', '') for c in topic_cols]
            active = [(n, v) for n, v in zip(t_names, t_vals) if v > 0]
            if active:
                fig_t = go.Figure(go.Bar(x=[a[0] for a in active], y=[a[1] for a in active],
                    marker_color=['#3b82f6','#10b981','#f59e0b','#a855f7','#ef4444','#06b6d4','#f97316','#84cc16','#ec4899','#6366f1'][:len(active)],
                    text=[f"{v:.2f}" for _, v in active], textposition='auto'))
                fig_t.update_layout(**PLOTLY_TEMPLATE['layout'])
                fig_t.update_layout(height=300, title="Topic Distribution")
                st.plotly_chart(fig_t, use_container_width=True)

    else:
        st.info("No NLP signals. Run NLP training pipeline first.")

    # ── NLP Label Quality ──
    lq = load_nlp_label_quality()
    if lq:
        st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-top:24px;margin-bottom:12px;">📊 NLP Label Quality Report</div>', unsafe_allow_html=True)

        # Named label mappings
        label_names = {
            'sentiment': {0: 'Negative', 1: 'Neutral', 2: 'Positive'},
            'event': {0: 'Policy', 1: 'Tech', 2: 'Supply', 3: 'Crash', 4: 'Launch', 5: 'Regulation', 6: 'Economic', 7: 'None'},
            'topic': {0: 'AI', 1: 'EV', 2: 'Semiconductors', 3: 'Crypto', 4: 'Climate', 5: 'Energy', 6: 'Healthcare', 7: 'Finance', 8: 'Geopolitics', 9: 'Other'},
            'entity': {0: 'Company', 1: 'Person', 2: 'Product', 3: 'Location', 4: 'Other'},
        }
        colors_map = {
            'sentiment': ['#ef4444', '#a1a1aa', '#10b981'],
            'event': ['#3b82f6','#06b6d4','#f59e0b','#ef4444','#10b981','#a855f7','#f97316','#71717a'],
            'topic': ['#3b82f6','#10b981','#f59e0b','#a855f7','#ef4444','#06b6d4','#f97316','#84cc16','#ec4899','#6366f1'],
            'entity': ['#3b82f6','#10b981','#f59e0b','#a855f7','#ef4444'],
        }

        # 2x2 grid of quality charts
        tasks = list(lq.items())
        for row_start in range(0, len(tasks), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if row_start + j < len(tasks):
                    task_name, info = tasks[row_start + j]
                    with col:
                        if isinstance(info, dict) and 'distribution' in info:
                            dist = info['distribution']
                            names_map = label_names.get(task_name, {})
                            task_colors = colors_map.get(task_name, ['#3b82f6'] * 20)

                            labels = [names_map.get(int(k), f'Class {k}') for k in dist.keys()]
                            values = list(dist.values())

                            entropy = info.get('normalized_entropy', 0)
                            balanced = info.get('is_balanced', False)
                            bal_label = '✅ Balanced' if balanced else '⚠️ Imbalanced'
                            bal_color = '#10b981' if balanced else '#f59e0b'

                            fig_q = go.Figure(go.Bar(x=labels, y=values,
                                marker_color=task_colors[:len(labels)],
                                text=[f"{v:,}" for v in values], textposition='auto'))
                            fig_q.update_layout(**PLOTLY_TEMPLATE['layout'])
                            fig_q.update_layout(height=300,
                                title=f"{task_name.title()} <span style='font-size:0.7rem;color:{bal_color};'>{bal_label} · Entropy: {entropy:.2f}</span>")
                            st.plotly_chart(fig_q, use_container_width=True)
                        else:
                            st.markdown(f'<div class="bento-card" style="padding:12px;"><strong>{task_name.title()}</strong><pre style="color:#a1a1aa;font-size:0.75rem;">{info}</pre></div>', unsafe_allow_html=True)

    # ── News Articles Feed ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:16px;">📰 Collected News Articles</div>', unsafe_allow_html=True)

    news_df = load_news_articles()
    if news_df is not None and not news_df.empty:
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Articles</div><div style="font-size:1.1rem;font-weight:700;color:#f4f4f5;">{len(news_df):,}</div></div>', unsafe_allow_html=True)
        with nc2:
            sources_n = news_df['source'].nunique() if 'source' in news_df.columns else 0
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Sources</div><div style="font-size:1.1rem;font-weight:700;color:#3b82f6;">{sources_n}</div></div>', unsafe_allow_html=True)
        with nc3:
            date_range_n = ''
            if 'published_at' in news_df.columns:
                min_d = news_df['published_at'].min()
                max_d = news_df['published_at'].max()
                date_range_n = f"{min_d.strftime('%Y-%m-%d') if pd.notna(min_d) else '?'} → {max_d.strftime('%Y-%m-%d') if pd.notna(max_d) else '?'}"
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Date Range</div><div style="font-size:0.85rem;font-weight:600;color:#f4f4f5;">{date_range_n}</div></div>', unsafe_allow_html=True)

        # Source distribution chart
        if 'source' in news_df.columns:
            src_dist = news_df['source'].value_counts().head(10)
            fig_ns = go.Figure(go.Bar(x=src_dist.values, y=src_dist.index, orientation='h',
                marker_color='#3b82f6', text=src_dist.values, textposition='auto'))
            fig_ns.update_layout(**PLOTLY_TEMPLATE['layout'])
            fig_ns.update_layout(height=280, title="Top News Sources")
            st.plotly_chart(fig_ns, use_container_width=True)

        # Article table
        show_cols_n = [c for c in ['title', 'source', 'author', 'published_at'] if c in news_df.columns]
        if show_cols_n:
            st.dataframe(news_df[show_cols_n].head(30), use_container_width=True, hide_index=True, height=250)
    else:
        st.info("No news articles. Run data collection first.")

    # ── Research Papers (arXiv) ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:16px;">📄 Research Papers (arXiv)</div>', unsafe_allow_html=True)

    research_df = load_research_papers()
    if research_df is not None and not research_df.empty:
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Papers Collected</div><div style="font-size:1.1rem;font-weight:700;color:#f4f4f5;">{len(research_df):,}</div></div>', unsafe_allow_html=True)
        with rc2:
            cats = research_df['category'].nunique() if 'category' in research_df.columns else 0
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Categories</div><div style="font-size:1.1rem;font-weight:700;color:#a855f7;">{cats}</div></div>', unsafe_allow_html=True)

        if 'category' in research_df.columns:
            cat_dist_r = research_df['category'].value_counts().head(12)
            fig_rc = go.Figure(go.Bar(x=cat_dist_r.index, y=cat_dist_r.values,
                marker_color=['#3b82f6','#10b981','#f59e0b','#a855f7','#ef4444','#06b6d4','#f97316','#84cc16','#ec4899','#6366f1','#14b8a6','#f43f5e'][:len(cat_dist_r)],
                text=cat_dist_r.values, textposition='auto'))
            fig_rc.update_layout(**PLOTLY_TEMPLATE['layout'])
            fig_rc.update_layout(height=300, title="Research by Category")
            st.plotly_chart(fig_rc, use_container_width=True)

        # Paper table
        show_cols_r = [c for c in ['title', 'category', 'published_date', 'authors'] if c in research_df.columns]
        if show_cols_r:
            display_r = research_df[show_cols_r].copy()
            if 'title' in display_r.columns:
                display_r['title'] = display_r['title'].str[:80]
            if 'authors' in display_r.columns:
                display_r['authors'] = display_r['authors'].str[:40]
            st.dataframe(display_r.head(30), use_container_width=True, hide_index=True, height=250)
    else:
        st.info("No research papers. Run data collection first.")

    # ── NASA Patents ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:16px;">🔬 NASA Patents & Innovation</div>', unsafe_allow_html=True)

    patent_df = load_patent_data()
    if patent_df is not None and not patent_df.empty:
        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Patents</div><div style="font-size:1.1rem;font-weight:700;color:#f4f4f5;">{len(patent_df):,}</div></div>', unsafe_allow_html=True)
        with pc2:
            centers = patent_df['origin_center'].nunique() if 'origin_center' in patent_df.columns else 0
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Research Centers</div><div style="font-size:1.1rem;font-weight:700;color:#10b981;">{centers}</div></div>', unsafe_allow_html=True)

        pp1, pp2 = st.columns(2)
        with pp1:
            if 'category' in patent_df.columns:
                p_cat = patent_df['category'].value_counts().head(8)
                fig_pc = go.Figure(go.Pie(labels=p_cat.index, values=p_cat.values, hole=0.4,
                    textinfo='label+percent'))
                fig_pc.update_layout(**PLOTLY_TEMPLATE['layout'])
                fig_pc.update_layout(height=300, title="Patents by Category")
                st.plotly_chart(fig_pc, use_container_width=True)
        with pp2:
            if 'origin_center' in patent_df.columns:
                p_center = patent_df['origin_center'].value_counts().head(8)
                fig_po = go.Figure(go.Bar(x=p_center.values, y=p_center.index, orientation='h',
                    marker_color='#10b981', text=p_center.values, textposition='auto'))
                fig_po.update_layout(**PLOTLY_TEMPLATE['layout'])
                fig_po.update_layout(height=300, title="Patents by Center")
                st.plotly_chart(fig_po, use_container_width=True)

        show_cols_p = [c for c in ['title', 'category', 'origin_center', 'search_keyword'] if c in patent_df.columns]
        if show_cols_p:
            display_p = patent_df[show_cols_p].copy()
            if 'title' in display_p.columns:
                display_p['title'] = display_p['title'].str[:80]
            st.dataframe(display_p.head(30), use_container_width=True, hide_index=True, height=250)
    else:
        st.info("No patent data. Run data collection first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ALTERNATIVE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def page_alternative_data():
    render_section_header("Alternative Data Intelligence", "Real alternative data indices from global sources")

    st.markdown('''
    <div style="background: rgba(16,185,129,0.06); border: 1px solid #27272a; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
        <div style="color:#a1a1aa; font-size: 0.78rem; line-height: 1.5;">
            🌎 <strong style="color:#f4f4f5;">Why Alternative Data Matters:</strong>
            Traditional stock models only see price/volume. This platform enriches predictions with
            <strong style="color:#10b981;">macro-economic indicators</strong> (GDP, CPI, interest rates via FRED),
            <strong style="color:#f59e0b;">energy supply data</strong> (oil/gas/coal production via EIA),
            <strong style="color:#3b82f6;">global weather patterns</strong> (affecting agriculture & energy demand),
            <strong style="color:#a855f7;">job market trends</strong> (consumer spending proxy),
            and <strong style="color:#ef4444;">trade flows</strong> (international economic health).
            These signals capture macro-level forces that move markets beyond what price charts alone reveal.
        </div>
    </div>
    ''', unsafe_allow_html=True)

    alt_data = get_alternative_data_summary()
    if not alt_data:
        st.info("No alternative data indices found.")
        return

    # KPI grid
    kpi_html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;">'
    for item in alt_data:
        kpi_html += f'''<div class="bento-card" style="padding:12px;text-align:center;">
            <div style="font-size:1.5rem;">{item["icon"]}</div>
            <div style="color:#a1a1aa;font-size:0.7rem;">{item["name"]}</div>
            <div style="font-size:1.1rem;font-weight:600;color:#f4f4f5;">{item["records"]:,} pts</div>
            <div style="color:#71717a;font-size:0.65rem;">{item["date_range"]}</div></div>'''
    kpi_html += '</div>'
    st.markdown(kpi_html, unsafe_allow_html=True)

    # Charts — 2 per row, handle multi-country data
    chartable = [item for item in alt_data if item['records'] >= 2]  # Skip single-point indices
    for i in range(0, len(chartable), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(chartable):
                item = chartable[i + j]
                with col:
                    df = load_alternative_data_index(item['file'])
                    if df is not None and 'date' in df.columns:
                        exclude_cols = {'date', 'country', 'country_code', 'origin_country'}
                        val_cols = [c for c in df.columns if c not in exclude_cols]
                        country_col = next((c for c in ['country', 'origin_country', 'country_code'] if c in df.columns), None)

                        fig = go.Figure()
                        if country_col and df[country_col].nunique() > 1:
                            # Multi-country: aggregate by date or show top countries
                            top_countries = df.groupby(country_col)[val_cols[0]].sum().nlargest(5).index.tolist() if val_cols else []
                            for cn in top_countries:
                                cn_data = df[df[country_col] == cn].sort_values('date')
                                if not cn_data.empty and val_cols:
                                    fig.add_trace(go.Scatter(x=cn_data['date'], y=cn_data[val_cols[0]],
                                        name=str(cn)[:15], line=dict(width=1.5)))
                        else:
                            # Single-country or no country column
                            for vc in val_cols[:2]:
                                fig.add_trace(go.Scatter(x=df['date'], y=df[vc],
                                    name=vc.replace('_', ' ').title(), line=dict(width=1.5),
                                    fill='tozeroy' if len(val_cols) == 1 else None,
                                    fillcolor='rgba(59,130,246,0.1)' if len(val_cols) == 1 else None))

                        fig.update_layout(**PLOTLY_TEMPLATE['layout'])
                        fig.update_layout(height=300, title=f"{item['icon']} {item['name']}", showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)

    # Single-point indices — show as stat cards instead of charts
    single_pt = [item for item in alt_data if item['records'] < 2]
    if single_pt:
        st.markdown('<div class="card-title" style="margin-top:16px;margin-bottom:8px;">📌 Single-Point Indices</div>', unsafe_allow_html=True)
        sp_html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;">'
        for item in single_pt:
            df = load_alternative_data_index(item['file'])
            if df is not None:
                exclude_cols = {'date', 'country', 'country_code', 'origin_country'}
                val_cols = [c for c in df.columns if c not in exclude_cols]
                val = df[val_cols[0]].iloc[0] if val_cols else 'N/A'
                try:
                    val_display = f"{float(val):.2f}"
                except (ValueError, TypeError):
                    val_display = str(val)
                sp_html += f'''<div class="bento-card" style="padding:10px;text-align:center;">
                    <div style="font-size:1.2rem;">{item["icon"]}</div>
                    <div style="color:#a1a1aa;font-size:0.65rem;">{item["name"]}</div>
                    <div style="font-size:1rem;font-weight:600;color:#f4f4f5;">{val_display}</div>
                    <div style="color:#71717a;font-size:0.6rem;">{item["date_range"]}</div></div>'''
        sp_html += '</div>'
        st.markdown(sp_html, unsafe_allow_html=True)

    # Crypto
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:16px;">₿ Cryptocurrency Data (100 coins)</div>', unsafe_allow_html=True)

    crypto_df = load_crypto_data()
    if crypto_df is not None and not crypto_df.empty:
        coins = sorted(crypto_df['coin'].unique()) if 'coin' in crypto_df.columns else []
        top_coins = [c for c in ['bitcoin', 'ethereum', 'solana', 'cardano', 'dogecoin'] if c in coins]
        selected_coins = st.multiselect("Select Coins", coins, default=top_coins[:3] if top_coins else coins[:3])
        if selected_coins:
            filtered = crypto_df[crypto_df['coin'].isin(selected_coins)]
            date_col_c = 'date' if 'date' in filtered.columns else 'timestamp'

            cc1, cc2 = st.columns([2, 1])
            with cc1:
                fig_c = go.Figure()
                for coin in selected_coins:
                    coin_data = filtered[filtered['coin'] == coin].sort_values(date_col_c)
                    fig_c.add_trace(go.Scatter(x=coin_data[date_col_c], y=coin_data['price'], name=coin.title(), line=dict(width=1.5)))
                fig_c.update_layout(**PLOTLY_TEMPLATE['layout'])
                fig_c.update_layout(height=400, yaxis_title="Price (USD)")
                st.plotly_chart(fig_c, use_container_width=True)
            with cc2:
                # Crypto summary table
                summary_rows = []
                for coin in selected_coins:
                    coin_data = filtered[filtered['coin'] == coin]
                    if not coin_data.empty and 'price' in coin_data.columns:
                        summary_rows.append({
                            'Coin': coin.title(),
                            'Price': f"${coin_data['price'].iloc[-1]:,.2f}",
                            'Market Cap': f"${coin_data['market_cap'].iloc[-1]:,.0f}" if 'market_cap' in coin_data.columns else 'N/A',
                        })
                if summary_rows:
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No crypto data found.")

    # ── FRED Economic Dashboard ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:4px;">🏛 FRED Economic Dashboard (47 Indicators)</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#71717a;font-size:0.72rem;margin-bottom:12px;">Federal Reserve Economic Data — macroeconomic indicators that drive market sentiment. GDP growth signals economic expansion, CPI tracks inflation, the VIX measures market fear, and interest rates affect corporate valuations. These are key inputs to our regime-adaptive ensemble.</div>', unsafe_allow_html=True)

    fred_df = load_economic_indicators()
    if fred_df is not None and not fred_df.empty and 'indicator' in fred_df.columns:
        indicators = sorted(fred_df['indicator'].unique())
        # Show key macro KPIs first
        key_indicators = ['GDP', 'CPI', 'UNEMPLOYMENT_RATE', 'INTEREST_RATE', 'VIX_INDEX', 'CRUDE_OIL_PRICE']
        avail_key = [k for k in key_indicators if k in indicators]
        if avail_key:
            kpi_cols = st.columns(len(avail_key))
            for i, ind in enumerate(avail_key):
                with kpi_cols[i]:
                    ind_data = fred_df[fred_df['indicator'] == ind].sort_values('date')
                    latest_val = ind_data['value'].iloc[-1] if not ind_data.empty else 0
                    latest_date = ind_data['date'].iloc[-1].strftime('%Y-%m') if not ind_data.empty else 'N/A'
                    st.markdown(f'''<div class="bento-card" style="padding:10px;text-align:center;">
                        <div style="color:#a1a1aa;font-size:0.65rem;">{ind.replace('_', ' ').title()}</div>
                        <div style="font-size:1.1rem;font-weight:700;color:#f4f4f5;">{latest_val:,.2f}</div>
                        <div style="color:#71717a;font-size:0.6rem;">{latest_date}</div></div>''', unsafe_allow_html=True)

        # Interactive selector for any indicator
        selected_indicators = st.multiselect("Select Indicators to Chart", indicators,
            default=[i for i in ['GDP', 'CPI', 'UNEMPLOYMENT_RATE'] if i in indicators][:3])
        if selected_indicators:
            fig_fred = go.Figure()
            colors_fred = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#a855f7', '#06b6d4']
            for idx, ind in enumerate(selected_indicators):
                ind_data = fred_df[fred_df['indicator'] == ind].sort_values('date')
                fig_fred.add_trace(go.Scatter(x=ind_data['date'], y=ind_data['value'],
                    name=ind.replace('_', ' ').title(), line=dict(width=1.5, color=colors_fred[idx % len(colors_fred)])))
            fig_fred.update_layout(**PLOTLY_TEMPLATE['layout'])
            fig_fred.update_layout(height=400, yaxis_title="Value", hovermode='x unified')
            st.plotly_chart(fig_fred, use_container_width=True)
    else:
        st.info("No economic indicators data. Run data collection first.")

    # ── Energy Analytics ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:4px;">⚡ Energy Analytics (EIA)</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#71717a;font-size:0.72rem;margin-bottom:12px;">U.S. Energy Information Administration data tracking oil, gas, coal, and electricity production/demand. Energy prices directly impact transportation, manufacturing, and consumer costs — making them a leading indicator of inflation and corporate earnings.</div>', unsafe_allow_html=True)

    energy_df = load_energy_data()
    if energy_df is not None and not energy_df.empty and 'series' in energy_df.columns:
        energy_series = sorted(energy_df['series'].unique())
        energy_colors = {'electricity_demand': '#f59e0b', 'crude_oil_production': '#3b82f6',
                         'natural_gas_production': '#10b981', 'coal_production': '#ef4444'}

        e_cols = st.columns(len(energy_series))
        for i, series in enumerate(energy_series):
            with e_cols[i]:
                s_data = energy_df[energy_df['series'] == series]
                latest = s_data['value'].iloc[-1] if not s_data.empty else 0
                st.markdown(f'''<div class="bento-card" style="padding:10px;text-align:center;border-top:2px solid {energy_colors.get(series, "#3b82f6")};">
                    <div style="color:#a1a1aa;font-size:0.65rem;">{series.replace('_', ' ').title()}</div>
                    <div style="font-size:1rem;font-weight:700;color:#f4f4f5;">{latest:,.1f}</div>
                    <div style="color:#71717a;font-size:0.6rem;">{s_data['unit'].iloc[0] if 'unit' in s_data.columns and not s_data.empty else ''}</div></div>''', unsafe_allow_html=True)

        fig_energy = go.Figure()
        for series in energy_series:
            s_data = energy_df[energy_df['series'] == series].sort_values('date')
            fig_energy.add_trace(go.Scatter(x=s_data['date'], y=s_data['value'],
                name=series.replace('_', ' ').title(),
                line=dict(width=1.5, color=energy_colors.get(series, '#3b82f6'))))
        fig_energy.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig_energy.update_layout(height=350, yaxis_title="Production/Demand", hovermode='x unified')
        st.plotly_chart(fig_energy, use_container_width=True)
    else:
        st.info("No energy data. Run data collection first.")

    # ── Weather Intelligence ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:4px;">🌤 Global Weather Intelligence (101 Cities)</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#71717a;font-size:0.72rem;margin-bottom:12px;">Real-time weather data across 101 major cities worldwide. Extreme temperatures affect energy demand, agricultural yields, and supply chains — all of which can move markets. Temperature, humidity, and wind speed are tracked as macro-context features.</div>', unsafe_allow_html=True)

    weather_df = load_weather_data()
    if weather_df is not None and not weather_df.empty:
        w_cols = st.columns(4)
        with w_cols[0]:
            avg_temp = weather_df['temperature'].mean() if 'temperature' in weather_df.columns else 0
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Avg Temperature</div><div style="font-size:1.1rem;font-weight:700;color:#f59e0b;">{avg_temp:.1f}°C</div></div>', unsafe_allow_html=True)
        with w_cols[1]:
            avg_humid = weather_df['humidity'].mean() if 'humidity' in weather_df.columns else 0
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Avg Humidity</div><div style="font-size:1.1rem;font-weight:700;color:#3b82f6;">{avg_humid:.0f}%</div></div>', unsafe_allow_html=True)
        with w_cols[2]:
            cities = weather_df['city'].nunique() if 'city' in weather_df.columns else 0
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Cities Tracked</div><div style="font-size:1.1rem;font-weight:700;color:#10b981;">{cities}</div></div>', unsafe_allow_html=True)
        with w_cols[3]:
            avg_wind = weather_df['wind_speed'].mean() if 'wind_speed' in weather_df.columns else 0
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Avg Wind Speed</div><div style="font-size:1.1rem;font-weight:700;color:#a855f7;">{avg_wind:.1f} m/s</div></div>', unsafe_allow_html=True)

        # Temperature bar chart by city (top 15 hottest)
        if 'temperature' in weather_df.columns and 'city' in weather_df.columns:
            sorted_w = weather_df.sort_values('temperature', ascending=False).head(15)
            fig_w = go.Figure(go.Bar(
                x=sorted_w['city'], y=sorted_w['temperature'],
                marker_color=[f'hsl({max(0, 60 - t)}, 80%, 55%)' for t in sorted_w['temperature']],
                text=[f"{t:.1f}°C" for t in sorted_w['temperature']], textposition='auto'))
            fig_w.update_layout(**PLOTLY_TEMPLATE['layout'])
            fig_w.update_layout(height=300, yaxis_title="Temperature (°C)", title="Top 15 Hottest Cities")
            st.plotly_chart(fig_w, use_container_width=True)

        # Data table
        display_cols = [c for c in ['city', 'temperature', 'humidity', 'pressure', 'wind_speed', 'weather'] if c in weather_df.columns]
        st.dataframe(weather_df[display_cols].sort_values('city'), use_container_width=True, hide_index=True, height=250)
    else:
        st.info("No weather data. Run data collection first.")

    # ── Job Market Intelligence ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:4px;">💼 Job Market Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#71717a;font-size:0.72rem;margin-bottom:12px;">Aggregated job listings from Adzuna and USAJobs APIs. Employment trends are a leading indicator of consumer spending power and economic health — rising tech hiring often precedes sector growth.</div>', unsafe_allow_html=True)

    jobs_df = load_jobs_data()
    if jobs_df is not None and not jobs_df.empty:
        j_cols = st.columns(3)
        with j_cols[0]:
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Total Listings</div><div style="font-size:1.1rem;font-weight:700;color:#f4f4f5;">{len(jobs_df):,}</div></div>', unsafe_allow_html=True)
        with j_cols[1]:
            sources_count = jobs_df['source'].nunique() if 'source' in jobs_df.columns else 0
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Data Sources</div><div style="font-size:1.1rem;font-weight:700;color:#3b82f6;">{sources_count}</div></div>', unsafe_allow_html=True)
        with j_cols[2]:
            countries = jobs_df['country'].nunique() if 'country' in jobs_df.columns else jobs_df['location'].nunique() if 'location' in jobs_df.columns else 0
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Regions</div><div style="font-size:1.1rem;font-weight:700;color:#10b981;">{countries}</div></div>', unsafe_allow_html=True)

        jc1, jc2 = st.columns(2)
        with jc1:
            # Category distribution
            cat_col = 'category' if 'category' in jobs_df.columns else 'job_category' if 'job_category' in jobs_df.columns else None
            if cat_col:
                cat_dist = jobs_df[cat_col].value_counts().head(10)
                fig_jc = go.Figure(go.Bar(x=cat_dist.values, y=cat_dist.index, orientation='h',
                    marker_color='#3b82f6', text=cat_dist.values, textposition='auto'))
                fig_jc.update_layout(**PLOTLY_TEMPLATE['layout'])
                fig_jc.update_layout(height=300, title="Top Job Categories")
                st.plotly_chart(fig_jc, use_container_width=True)
        with jc2:
            # Salary distribution
            sal_col = None
            if 'salary_max' in jobs_df.columns:
                sal_col = 'salary_max'
            elif 'salary_range' in jobs_df.columns:
                sal_col = 'salary_range'
            if sal_col and jobs_df[sal_col].dtype in ['float64', 'int64']:
                fig_js = go.Figure(go.Histogram(x=jobs_df[sal_col].dropna(), nbinsx=30, marker_color='#10b981'))
                fig_js.update_layout(**PLOTLY_TEMPLATE['layout'])
                fig_js.update_layout(height=300, title="Salary Distribution", xaxis_title="Salary (USD)")
                st.plotly_chart(fig_js, use_container_width=True)
            else:
                # Show location breakdown instead
                loc_col = 'location' if 'location' in jobs_df.columns else 'country'
                if loc_col in jobs_df.columns:
                    loc_dist = jobs_df[loc_col].value_counts().head(10)
                    fig_jl = go.Figure(go.Pie(labels=loc_dist.index, values=loc_dist.values, hole=0.5,
                        textinfo='label+percent'))
                    fig_jl.update_layout(**PLOTLY_TEMPLATE['layout'])
                    fig_jl.update_layout(height=300, title="Jobs by Location")
                    st.plotly_chart(fig_jl, use_container_width=True)

        # Job listings table
        show_cols = [c for c in ['job_title', 'company', 'organization', 'location', 'country', 'category', 'job_category', 'salary_range', 'source'] if c in jobs_df.columns]
        st.dataframe(jobs_df[show_cols].head(50), use_container_width=True, hide_index=True, height=250)
    else:
        st.info("No jobs data. Run data collection first.")

    # ── Global Trade Data ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:4px;">🌍 Global Trade Data (World Bank)</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#71717a;font-size:0.72rem;margin-bottom:12px;">International trade volumes and indicators from the World Bank. Trade flows reflect global economic health — declining exports often signal upcoming recessions, while trade surpluses indicate economic strength.</div>', unsafe_allow_html=True)

    trade_df = load_trade_data()
    if trade_df is not None and not trade_df.empty:
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Records</div><div style="font-size:1.1rem;font-weight:700;color:#f4f4f5;">{len(trade_df):,}</div></div>', unsafe_allow_html=True)
        with tc2:
            countries_t = trade_df['country'].nunique() if 'country' in trade_df.columns else 0
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Countries</div><div style="font-size:1.1rem;font-weight:700;color:#3b82f6;">{countries_t}</div></div>', unsafe_allow_html=True)
        with tc3:
            indicators_t = trade_df['indicator'].nunique() if 'indicator' in trade_df.columns else 0
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Indicators</div><div style="font-size:1.1rem;font-weight:700;color:#10b981;">{indicators_t}</div></div>', unsafe_allow_html=True)

        if 'country' in trade_df.columns and 'value' in trade_df.columns:
            top_trade = trade_df.groupby('country')['value'].mean().nlargest(12).reset_index()
            fig_trade = go.Figure(go.Bar(x=top_trade['country'], y=top_trade['value'],
                marker_color='#3b82f6', text=[f"{v:,.1f}" for v in top_trade['value']], textposition='auto'))
            fig_trade.update_layout(**PLOTLY_TEMPLATE['layout'])
            fig_trade.update_layout(height=300, title="Top 12 Countries by Trade Value")
            st.plotly_chart(fig_trade, use_container_width=True)
    else:
        st.info("No trade data. Run data collection first.")

    # ── Population Trends ──
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:4px;">👥 Population Trends (UN Data)</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#71717a;font-size:0.72rem;margin-bottom:12px;">United Nations population statistics for 20 major economies. Population growth drives long-term demand for goods, services, and housing — affecting GDP trajectory and market valuations at a structural level.</div>', unsafe_allow_html=True)

    pop_df = load_population_data()
    if pop_df is not None and not pop_df.empty:
        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Records</div><div style="font-size:1.1rem;font-weight:700;color:#f4f4f5;">{len(pop_df):,}</div></div>', unsafe_allow_html=True)
        with pc2:
            countries_p = pop_df['country'].nunique() if 'country' in pop_df.columns else 0
            st.markdown(f'<div class="bento-card" style="padding:10px;text-align:center;"><div style="color:#a1a1aa;font-size:0.65rem;">Countries</div><div style="font-size:1.1rem;font-weight:700;color:#3b82f6;">{countries_p}</div></div>', unsafe_allow_html=True)

        if 'country' in pop_df.columns and 'value' in pop_df.columns:
            top_pop = pop_df.groupby('country')['value'].max().nlargest(15).reset_index()
            fig_pop = go.Figure(go.Bar(x=top_pop['country'], y=top_pop['value'],
                marker_color='#a855f7', text=[f"{v:,.0f}" for v in top_pop['value']], textposition='auto'))
            fig_pop.update_layout(**PLOTLY_TEMPLATE['layout'])
            fig_pop.update_layout(height=300, title="Top 15 Countries by Population")
            st.plotly_chart(fig_pop, use_container_width=True)
    else:
        st.info("No population data. Run data collection first.")

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

PAGES = {
    "🏠 Overview": page_overview,
    "📡 Data Sources": page_data_sources,
    "⚙️ Data Pipeline": page_pipeline,
    "📊 Datasets": page_datasets,
    "🤖 Model Training": page_model_training,
    "📈 Model Evaluation": page_model_evaluation,
    "🔮 Predictions": page_predictions,
    "🚨 Anomaly & Regimes": page_anomaly_detection,
    "📱 Social & NLP": page_social_intelligence,
    "🌍 Alternative Data": page_alternative_data,
    "📉 Visualization": page_visualization,
    "💻 System Monitor": page_system_monitor,
    "📋 Logs": page_logs,
    "🛠 Settings": page_settings,
}

if page in PAGES:
    PAGES[page]()
else:
    st.error(f"Page '{page}' not found.")
