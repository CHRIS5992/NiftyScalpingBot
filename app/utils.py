"""
utils.py — Shared helper functions.
"""
from __future__ import annotations


def inject_custom_css() -> str:
    """Return custom CSS for Bloomberg-dark institutional theme."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Hide default Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Metric Cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        border-color: #FFD700;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* ── Section Headers ── */
    .gradient-header {
        background: linear-gradient(90deg, #FFD700 0%, #FF9800 50%, #FF5722 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .sub-header {
        color: #8b949e;
        font-size: 0.95rem;
        font-weight: 400;
        margin-bottom: 24px;
    }

    /* ── Cards ── */
    .info-card {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .info-card h4 {
        color: #FFD700;
        margin-bottom: 8px;
        font-weight: 700;
    }
    .info-card p, .info-card li {
        color: #c9d1d9;
        line-height: 1.7;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #161b22;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #FFD700 !important;
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border: 1px solid #21262d;
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: #161b22;
        border-radius: 8px;
        font-weight: 600;
    }

    /* ── Divider ── */
    hr {
        border-color: #21262d;
    }

    /* ── Positive/Negative indicators ── */
    .metric-positive { color: #00E676 !important; }
    .metric-negative { color: #FF5252 !important; }
    .metric-neutral  { color: #FFD700 !important; }
    </style>
    """
