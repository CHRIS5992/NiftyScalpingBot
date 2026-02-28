"""
utils.py — Shared helper functions.
"""
from __future__ import annotations


def inject_custom_css() -> str:
    """Return custom CSS — clean institutional dark theme."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Metric Cards ── */
    div[data-testid="stMetric"] {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 18px;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #475569;
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }

    /* ── Header ── */
    .gradient-header {
        color: #F1F5F9;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 4px;
    }
    .sub-header {
        color: #64748B;
        font-size: 0.9rem;
        font-weight: 400;
        margin-bottom: 20px;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        border-bottom: 1px solid #334155;
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        padding: 10px 20px;
        font-weight: 500;
        font-size: 0.85rem;
        color: #94A3B8;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #60A5FA !important;
        border-bottom: 2px solid #60A5FA !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0F172A;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #F1F5F9 !important;
        font-weight: 600 !important;
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border: 1px solid #334155;
        border-radius: 6px;
        overflow: hidden;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: #1E293B;
        border-radius: 6px;
        font-weight: 500;
    }

    /* ── Divider ── */
    hr {
        border-color: #1E293B;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F1F5F9 !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
    }
    .stDownloadButton > button:hover {
        border-color: #60A5FA !important;
        color: #60A5FA !important;
    }

    /* ── Semantic colors ── */
    .metric-positive { color: #34D399 !important; }
    .metric-negative { color: #F87171 !important; }
    .metric-neutral  { color: #FBBF24 !important; }
    </style>
    """
