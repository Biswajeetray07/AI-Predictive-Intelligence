"""
Custom CSS for the AI Predictive Intelligence Dashboard.
Next.js / ShadCN / Linear inspired Minimalist SaaS theme.
"""

MAIN_CSS = """
<style>
    /* ── Import Google Fonts ────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500&display=swap');

    /* ── Root Variables (Zinc-950 Theme) ───────────────────── */
    :root {
        --bg-primary: #09090b; /* zinc-950 */
        --bg-secondary: #09090b;
        --bg-card: #18181b; /* zinc-900 */
        --bg-card-hover: #27272a; /* zinc-800 */
        --primary: #cbd5e1; /* slate-300 */
        --accent: #3b82f6; /* blue-500 soft */
        --success: #10b981; /* green-500 soft */
        --warning: #f59e0b; /* amber-500 soft */
        --danger: #ef4444; /* red-500 soft */
        --text-primary: #f4f4f5; /* zinc-100 */
        --text-secondary: #a1a1aa; /* zinc-400 */
        --border: #27272a; /* zinc-800 */
        --glow: none;
    }

    /* ── Global Overrides ─────────────────────────────────── */
    .stApp {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Geist', 'Inter', sans-serif !important;
    }

    header[data-testid="stHeader"] {
        background-color: transparent !important;
        border-bottom: none !important;
    }

    /* ── Sidebar ──────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-primary) !important;
        border-right: 1px solid var(--border) !important;
        padding-top: 1rem !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: var(--bg-primary) !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: var(--text-secondary) !important;
        font-weight: 500;
        padding: 10px 14px !important;
        border-radius: 6px;
        transition: all 0.2s ease;
        font-size: 0.9rem !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: var(--bg-card-hover) !important;
        color: var(--text-primary) !important;
    }

    section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
    section[data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Native Streamlit Metrics (Fallback) ──────────────── */
    [data-testid="stMetric"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important; /* rounded-xl */
        padding: 24px !important; /* p-6 */
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important; /* shadow-sm */
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: #3f3f46 !important; /* zinc-700 */
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.875rem !important; /* text-sm */
        font-weight: 500 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 1.875rem !important; /* text-3xl */
        margin-top: 4px !important;
        font-family: 'Geist', 'Inter', sans-serif !important;
    }

    /* ── Custom Bento Grid Cards (ShadCN style) ───────────── */
    .bento-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px; /* rounded-2xl */
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-sm */
        transition: all 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    .bento-card:hover {
        border-color: #3f3f46; /* hover:border-zinc-700 */
    }
    
    .bento-hero {
        background: linear-gradient(145deg, var(--bg-card) 0%, #1e1e24 100%);
        border: 1px solid #27272a;
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .card-title {
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.01em;
    }
    
    .card-subtitle {
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin: 0;
        font-weight: 400;
        margin-bottom: 16px;
    }

    .card-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 8px 0;
        letter-spacing: -0.02em;
    }

    .card-trend {
        font-size: 0.875rem;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .trend-up { color: var(--success); }
    .trend-down { color: var(--danger); }
    .trend-neutral { color: var(--text-secondary); }

    /* ── Custom Status Badges ─────────────────────────────── */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        gap: 6px;
        border: 1px solid transparent;
    }
    
    .badge-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
    }

    .badge-active {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border-color: rgba(16, 185, 129, 0.2);
    }
    .badge-active .badge-dot { background-color: var(--success); }

    .badge-error {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger);
        border-color: rgba(239, 68, 68, 0.2);
    }
    .badge-error .badge-dot { background-color: var(--danger); }
    
    .badge-warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning);
        border-color: rgba(245, 158, 11, 0.2);
    }
    .badge-warning .badge-dot { background-color: var(--warning); }
    
    .badge-neutral {
        background: rgba(161, 161, 170, 0.1);
        color: var(--text-secondary);
        border-color: var(--border);
    }
    .badge-neutral .badge-dot { background-color: var(--text-secondary); }

    /* ── Logs List ────────────────────────────────────────── */
    .log-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid var(--border);
        font-size: 0.875rem;
    }
    .log-item:last-child {
        border-bottom: none;
    }
    .log-time {
        color: var(--text-secondary);
        font-family: 'Geist Mono', monospace;
        font-size: 0.8rem;
    }
    .log-msg {
        color: var(--text-primary);
        flex-grow: 1;
        margin-left: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── Top Nav / Header overrides ───────────────────────── */
    .section-header {
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--border);
    }
    .dashboard-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin: 0;
    }
    .dashboard-subtitle {
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-top: 4px;
        margin-bottom: 0;
    }

    /* ── DataFrame & Expanders ────────────────────────────── */
    .stDataFrame {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stExpander"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }

    /* ── Hide Streamlit default branding ──────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Optimize column gaps for Bento grid */
    [data-testid="column"] {
        padding: 0 8px;
    }
</style>
"""

# ── Clean SaaS Plotly Template ─────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Geist, Inter, sans-serif', color='#a1a1aa', size=11),
        xaxis=dict(
            gridcolor='#27272a',
            zerolinecolor='#27272a',
            linecolor='#27272a',
            showgrid=True,
            gridwidth=1,
            griddash='dot',
        ),
        yaxis=dict(
            gridcolor='#27272a',
            zerolinecolor='#27272a',
            linecolor='#27272a',
            showgrid=True,
            gridwidth=1,
            griddash='dot',
        ),
        margin=dict(l=30, r=20, t=30, b=30),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a1a1aa', size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        colorway=['#f4f4f5', '#a1a1aa', '#3b82f6', '#10b981', '#f59e0b', '#ef4444'],
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#18181b",
            font_size=12,
            font_family="Geist, Inter",
            bordercolor="#27272a",
        )
    )
)

# Shared Colors
COLORS = {
    'primary': '#f4f4f5',
    'secondary': '#a1a1aa',
    'accent': '#3b82f6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'bg_card': '#18181b',
    'border': '#27272a',
}
