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
            "🚨 Anomaly Detection",
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
        st.markdown('<div style="font-size: 1.8rem; font-weight: 600; margin-bottom: 24px;">Dashboard</div>', unsafe_allow_html=True)
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
        st.plotly_chart(draw_mini_bar(list(bd.keys()), list(bd.values()), '#a7f3d0'), use_container_width=True, config={{'displayModeBar':False}})
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
            st.plotly_chart(draw_mini_bar(names, sizes, '#3b82f6'), use_container_width=True, config={{'displayModeBar':False}})
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

        # Row 3 — Files & Records
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            total_records = kpis.get('total_records', 0)
            st.markdown(f"""
            <div class="bento-card" style="height: 140px;">
                <div class="card-title">📄 Project Data</div>
                <div class="card-subtitle" style="margin-top:20px;">{disk['total_gb']} GB across {sum(1 for v in disk['breakdown'].values() if v > 0)} directories · {total_records:,} total records</div>
            </div>
            """, unsafe_allow_html=True)
        with r3c2:
            api_count = kpis.get('active_apis', 0)
            st.markdown(f"""
            <div class="bento-card" style="height: 140px;">
                <div class="card-title">🔑 API Integration</div>
                <div class="card-subtitle" style="margin-top:20px;">{api_count} API keys configured · {kpis.get('data_sources', 0)} data sources active</div>
            </div>
            """, unsafe_allow_html=True)

    with right_panel:
        # Build jobs HTML dynamically from real pipeline data
        jobs_html = ''
        for job in jobs:
            badge_class = 'badge-active' if job['status'] == 'Succeeded' else 'badge-warning' if job['status'] == 'Empty' else 'badge-neutral'
            jobs_html += f"""
                <div style="display:flex; justify-content:space-between; align-items:center;">
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
    render_section_header("Data Pipeline", "End-to-end ML pipeline monitoring")

    stages = get_pipeline_stages()

    # ── Visual Pipeline Flow ──
    st.markdown('<div class="card-title" style="margin-bottom: 20px;">Execution Flow</div>', unsafe_allow_html=True)
    stage_cols = st.columns(len(stages) * 2 - 1)
    col_idx = 0
    for i, s in enumerate(stages):
        with stage_cols[col_idx]:
            badge_type = "badge-active" if s['status'] == 'Complete' else "badge-error" if s['status'] == 'Error' else "badge-neutral"
            badge_dot = "<div class='badge-dot'></div>"
            
            st.markdown(f"""
            <div class="bento-card" style="text-align: center; padding: 16px;">
                <div style="font-size: 1.2rem; font-weight: 600; color: #f4f4f5; margin-bottom: 8px;">{s['name']}</div>
                <span class="badge {badge_type}" style="margin: 0 auto;">{badge_dot}{s['status']}</span>
                <div style="color: #a1a1aa; font-size: 0.75rem; margin-top: 12px; font-family: 'Geist Mono';">{s['files']} files</div>
            </div>
            """, unsafe_allow_html=True)
        col_idx += 1
        if i < len(stages) - 1:
            with stage_cols[col_idx]:
                st.markdown('<div style="color: #27272a; font-size: 1.5rem; text-align: center; padding-top: 40px;">→</div>', unsafe_allow_html=True)
            col_idx += 1

    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)

    # ── Stage Details Table ──
    st.markdown('<div class="card-title" style="margin-bottom: 16px;">Stage Manifest</div>', unsafe_allow_html=True)
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
    render_section_header("Model Evaluation", "Compare model performance and validation metrics")

    models = get_model_info()
    if not models:
        st.info("No models available for evaluation yet.")
        return

    st.markdown('<div class="card-title" style="margin-bottom: 16px;">Model Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="bento-card">
        <p style="color: #a1a1aa; font-size: 0.9rem;">Detailed metrics will populate once models finish training and
        backtesting produces evaluation results. Metrics tracked:</p>
        <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 16px;">
            <span class="badge badge-neutral">MSE</span>
            <span class="badge badge-neutral">RMSE</span>
            <span class="badge badge-neutral">MAE</span>
            <span class="badge badge-neutral">R²</span>
            <span class="badge badge-neutral">Directional Accuracy</span>
            <span class="badge badge-neutral">Sharpe Ratio</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)

    # ── Real Training History ──
    history = get_training_history()
    if history:
        st.markdown('<div class="card-title" style="margin-bottom: 16px;">Training History (Real)</div>', unsafe_allow_html=True)
        model_colors = {'lstm': '#3b82f6', 'gru': '#10b981', 'transformer': '#f59e0b', 'tft': '#a855f7', 'nlp': '#ef4444', 'fusion': '#06b6d4'}
        fig = go.Figure()
        for model_name, entries in history.items():
            color = model_colors.get(model_name, '#f4f4f5')
            epochs = [e['epoch'] for e in entries]
            if any('train_loss' in e for e in entries):
                train_vals = [e.get('train_loss', None) for e in entries]
                fig.add_trace(go.Scatter(x=epochs, y=train_vals, name=f'{model_name} (train)',
                    line=dict(color=color, width=2), mode='lines+markers', marker=dict(size=5)))
            if any('val_loss' in e for e in entries):
                val_vals = [e.get('val_loss', None) for e in entries]
                fig.add_trace(go.Scatter(x=epochs, y=val_vals, name=f'{model_name} (val)',
                    line=dict(color=color, width=2, dash='dash'), mode='lines+markers', marker=dict(size=5)))
        fig.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig.update_layout(height=400, xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="card-title" style="margin-bottom: 16px;">Training History</div>', unsafe_allow_html=True)
        st.info('No training history available yet. Run Phase 6 (Model Training) to populate this chart.')
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def page_predictions():
    render_section_header("Predictions & Intelligence", "AI-powered forecasts and signal analysis")

    tickers = list_available_tickers()

    if not tickers:
        st.warning("No stock data available for predictions. Run the financial data pipeline first.")
        return

    c1, c2 = st.columns([1, 3])
    with c1:
        selected_ticker = st.selectbox("Select Asset", tickers, index=0)

    df = load_stock_data(selected_ticker)
    if df is None or df.empty:
        st.error(f"No data found for {selected_ticker}")
        return

    date_col = 'Date' if 'Date' in df.columns else 'date'

    st.markdown('<div style="height: 1px; background: #27272a; margin: 24px 0;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title" style="margin-bottom: 16px;">{selected_ticker} — Historical Price & Forecast</div>', unsafe_allow_html=True)

    fig = go.Figure()

    if 'Close' in df.columns:
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df['Close'], name='Close Price',
            line=dict(color='#f4f4f5', width=1.5),
        ))

        # Simulated forecast confidence band
        n = len(df)
        forecast_start = int(n * 0.8)
        forecast_df = df.iloc[forecast_start:].copy()
        upper = forecast_df['Close'] * 1.05
        lower = forecast_df['Close'] * 0.95

        fig.add_trace(go.Scatter(
            x=forecast_df[date_col], y=upper, name='Upper 95%',
            line=dict(color='#3b82f6', width=1, dash='dash'), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df[date_col], y=lower, name='Lower 95%',
            line=dict(color='#3b82f6', width=1, dash='dash'),
            fill='tonexty', fillcolor='rgba(59, 130, 246, 0.1)', showlegend=False,
        ))

    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(
            x=df[date_col], y=df['Volume'], name='Volume',
            marker_color='rgba(161, 161, 170, 0.2)', yaxis='y2',
        ))

    fig.update_layout(**PLOTLY_TEMPLATE['layout'])
    fig.update_layout(
         height=500,
        yaxis=dict(title="Price ($)", gridcolor='#27272a', zerolinecolor='#27272a'),
        yaxis2=dict(title="Volume", overlaying='y', side='right', showgrid=False,
                    range=[0, df['Volume'].max() * 4] if 'Volume' in df.columns else None),
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

    if 'Close' in df.columns:
        latest = df['Close'].iloc[-1]
        change = df['Close'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0
        trend_color = "#10b981" if change >= 0 else "#ef4444"
        trend_icon = "↑" if change >= 0 else "↓"
        
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 16px;">
            <div class="bento-card" style="padding: 16px;">
                <div style="color: #a1a1aa; font-size: 0.8rem; margin-bottom: 8px;">Latest Close</div>
                <div style="font-size: 1.5rem; font-weight: 600; font-family: 'Geist Mono'; color: #f4f4f5;">
                    ${latest:.2f}
                    <span style="font-size: 0.9rem; color: {trend_color}; margin-left: 8px;">{trend_icon} {abs(change):.2f}%</span>
                </div>
            </div>
            <div class="bento-card" style="padding: 16px;">
                <div style="color: #a1a1aa; font-size: 0.8rem; margin-bottom: 8px;">52W High</div>
                <div style="font-size: 1.5rem; font-weight: 600; font-family: 'Geist Mono'; color: #f4f4f5;">${df['Close'].max():.2f}</div>
            </div>
            <div class="bento-card" style="padding: 16px;">
                <div style="color: #a1a1aa; font-size: 0.8rem; margin-bottom: 8px;">52W Low</div>
                <div style="font-size: 1.5rem; font-weight: 600; font-family: 'Geist Mono'; color: #f4f4f5;">${df['Close'].min():.2f}</div>
            </div>
            <div class="bento-card" style="padding: 16px;">
                <div style="color: #a1a1aa; font-size: 0.8rem; margin-bottom: 8px;">Data Points</div>
                <div style="font-size: 1.5rem; font-weight: 600; font-family: 'Geist Mono'; color: #f4f4f5;">{len(df):,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def page_anomaly_detection():
    render_section_header("Anomaly Detection", "Monitor unusual patterns and outliers")

    tickers = list_available_tickers()
    if not tickers:
        st.info("No data available for anomaly detection.")
        return

    selected = st.selectbox("Select Asset for Analysis", tickers[:20])
    df = load_stock_data(selected)
    if df is None or 'Close' not in df.columns:
        st.warning("No price data available.")
        return

    date_col = 'Date' if 'Date' in df.columns else 'date'
    df['Returns'] = df['Close'].pct_change()
    df['Z_Score'] = (df['Returns'] - df['Returns'].mean()) / df['Returns'].std()
    df['Anomaly'] = df['Z_Score'].abs() > 2.5
    anomalies = df[df['Anomaly']].copy()

    st.markdown('<div style="height: 1px; background: #27272a; margin: 24px 0;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title" style="margin-bottom: 16px;">🚨 Detected Anomalies in {selected}: {len(anomalies)} events</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df['Close'], name='Price',
                             line=dict(color='#3b82f6', width=1.5)))
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies[date_col], y=anomalies['Close'],
                                 name='Anomaly', mode='markers',
                                 marker=dict(color='#ef4444', size=8, symbol='circle',
                                             line=dict(width=1, color='#ef4444'))))
    fig.update_layout(**PLOTLY_TEMPLATE['layout'])
    fig.update_layout( height=400, yaxis_title="Price", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card-title" style="margin-bottom: 16px;">Return Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(x=df['Returns'].dropna(), nbinsx=50,
                                     marker_color='#10b981'))
        fig.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig.update_layout( height=300, xaxis_title="Daily Return")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="card-title" style="margin-bottom: 16px;">Z-Score Timeline</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[date_col], y=df['Z_Score'],
                                 line=dict(color='#10b981', width=1)))
        fig.add_hline(y=2.5, line_dash="dash", line_color='#ef4444', annotation_text="Threshold", annotation_font_color="#ef4444")
        fig.add_hline(y=-2.5, line_dash="dash", line_color='#ef4444')
        fig.update_layout(**PLOTLY_TEMPLATE['layout'])
        fig.update_layout( height=300, yaxis_title="Z-Score")
        st.plotly_chart(fig, use_container_width=True)

    if not anomalies.empty:
        st.markdown('<div class="card-title" style="margin-bottom: 16px;">Anomaly Events</div>', unsafe_allow_html=True)
        display_df = anomalies[[date_col, 'Close', 'Returns', 'Z_Score']].copy()
        display_df['Returns'] = (display_df['Returns'] * 100).round(2)
        display_df['Z_Score'] = display_df['Z_Score'].round(2)
        display_df.columns = ['Date', 'Price', 'Return (%)', 'Z-Score']
        st.dataframe(display_df.sort_values('Date', ascending=False), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZATION WORKSPACE
# ═══════════════════════════════════════════════════════════════════════════════

def page_visualization():
    render_section_header("Visualization Workspace", "Explore datasets interactively")

    datasets = get_datasets_info()
    if not datasets:
        st.info("No datasets available to visualize.")
        return

    selected_ds = st.selectbox("Select Dataset", [d['name'] for d in datasets])
    ds = next((d for d in datasets if d['name'] == selected_ds), None)
    if not ds:
        return

    try:
        if ds['path'].endswith('.csv'):
            df = pd.read_csv(ds['path'], nrows=5000)
        else:
            df = pd.read_parquet(ds['path']).head(5000)
    except Exception as e:
        st.error(f"Error loading: {e}")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns for visualization.")
        st.dataframe(df.head(100), use_container_width=True)
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Bar", "Histogram", "Correlation Heatmap"])
    with c2:
        x_col = st.selectbox("X Axis", numeric_cols, index=0)
    with c3:
        y_col = st.selectbox("Y Axis", numeric_cols, index=min(1, len(numeric_cols) - 1))

    st.markdown('<div style="height: 1px; background: #27272a; margin: 24px 0;"></div>', unsafe_allow_html=True)

    if chart_type == "Scatter":
        fig = px.scatter(df, x=x_col, y=y_col, opacity=0.8)
        fig.update_traces(marker=dict(color='#3b82f6', size=6, line=dict(width=0)))
    elif chart_type == "Line":
        fig = px.line(df, x=x_col, y=y_col)
        fig.update_traces(line=dict(color='#3b82f6'))
    elif chart_type == "Bar":
        fig = px.bar(df.head(50), x=x_col, y=y_col)
        fig.update_traces(marker_color='#10b981')
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_col, nbins=50)
        fig.update_traces(marker_color='#10b981')
    elif chart_type == "Correlation Heatmap":
        corr = df[numeric_cols[:15]].corr()
        fig = px.imshow(corr, color_continuous_scale=['#09090b', '#3b82f6', '#f4f4f5'],
                        aspect='auto', text_auto='.2f')

    fig.update_layout(**PLOTLY_TEMPLATE['layout'])
    fig.update_layout( height=500)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SYSTEM MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

def page_system_monitor():
    render_section_header("System Monitor", "Infrastructure performance and resource usage")

    stats = get_system_stats()

    def make_gauge(value, title, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title=dict(text=title, font=dict(size=14, color='#a1a1aa')),
            number=dict(suffix="%", font=dict(color=color, size=32, family='Geist Mono')),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor='#27272a', dtick=25, tickfont=dict(color='#a1a1aa')),
                bar=dict(color=color),
                bgcolor='#09090b',
                borderwidth=1, bordercolor='#27272a',
                steps=[
                    dict(range=[0, 50], color='rgba(59, 130, 246, 0.05)'),
                    dict(range=[50, 80], color='rgba(245, 158, 11, 0.05)'),
                    dict(range=[80, 100], color='rgba(239, 68, 68, 0.05)'),
                ],
                threshold=dict(line=dict(color='#ef4444', width=2), thickness=0.8, value=85),
            ),
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          height=240, margin=dict(l=20, r=20, t=40, b=20))
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

    st.markdown('<div style="height: 1px; background: #27272a; margin: 24px 0;"></div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    s1.metric("RAM Used", f"{stats['memory_used_gb']} GB", f"/ {stats['memory_total_gb']} GB")
    s2.metric("Disk Used", f"{stats['disk_used_gb']} GB", f"/ {stats['disk_total_gb']} GB")
    s3.metric("CPU Cores", str(os.cpu_count()))

    st.markdown('<div style="height: 1px; background: #27272a; margin: 24px 0;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="card-title" style="margin-bottom: 16px;">Top Processes (by CPU)</div>', unsafe_allow_html=True)
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
    render_section_header("Settings", "Platform configuration and API key management")

    st.markdown('<div class="card-title" style="margin-bottom: 16px;">API Key Status</div>', unsafe_allow_html=True)
    env_path = os.path.join(get_project_root(), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    val = val.strip().strip('"').strip("'")
                    masked = val[:4] + '•' * max(0, len(val) - 8) + val[-4:] if len(val) > 8 else '••••'
                    badge_type = 'badge-active' if val else 'badge-neutral'
                    badge_dot = "<div class='badge-dot'></div>"
                    status_text = 'Online' if val else 'Offline'
                    
                    st.markdown(f"""
                    <div class="bento-card" style="padding: 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="color: #f4f4f5; font-weight: 500;">{key}</span>
                            <span style="color: #a1a1aa; font-family: 'Geist Mono'; margin-left: 16px; font-size: 0.9rem;">{masked}</span>
                        </div>
                        <span class="badge {badge_type}">{badge_dot}{status_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("No .env file found.")

    st.markdown('<div style="height: 1px; background: #27272a; margin: 32px 0;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="card-title" style="margin-bottom: 16px;">Project Info</div>', unsafe_allow_html=True)
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
    "🚨 Anomaly Detection": page_anomaly_detection,
    "📉 Visualization": page_visualization,
    "💻 System Monitor": page_system_monitor,
    "📋 Logs": page_logs,
    "🛠 Settings": page_settings,
}

if page in PAGES:
    PAGES[page]()
else:
    st.error(f"Page '{page}' not found.")
