"""
utils.py — Shared helper functions & custom CSS (shadcn-inspired light theme).
"""
from __future__ import annotations


def inject_custom_css() -> str:
    """Return shadcn-inspired light-mode CSS for professional SaaS look."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Reset & Globals ── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: #FFFFFF;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Metric Cards (shadcn card style) ── */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(99,102,241,0.10);
        border-color: #C7D2FE;
    }
    div[data-testid="stMetric"] label {
        color: #6B7280 !important;
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] > svg { display: inline; }

    /* ── Headers ── */
    .gradient-header {
        color: #111827;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.4px;
        margin-bottom: 2px;
    }
    .sub-header {
        color: #9CA3AF;
        font-size: 0.85rem;
        font-weight: 400;
        margin-bottom: 20px;
    }

    /* ── Info Cards ── */
    .info-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
    }
    .info-card h4 {
        color: #111827;
        margin-bottom: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .info-card p, .info-card li {
        color: #4B5563;
        line-height: 1.7;
        font-size: 0.87rem;
    }

    /* ── Tabs (shadcn underline style) ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        padding: 10px 18px;
        font-weight: 500;
        font-size: 0.82rem;
        color: #6B7280;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
        transition: color 0.15s, border-color 0.15s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #111827;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #6366F1 !important;
        font-weight: 600;
        border-bottom-color: #6366F1 !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #F9FAFB;
        border-right: 1px solid #E5E7EB;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #111827 !important;
        font-weight: 600;
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: #F9FAFB;
        border-radius: 8px;
        font-weight: 600;
        color: #111827;
    }

    /* ── Divider ── */
    hr { border-color: #F3F4F6; }

    /* ── Buttons (shadcn primary) ── */
    .stDownloadButton > button,
    .stButton > button[kind="primary"] {
        background: #6366F1;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 16px;
        transition: background 0.15s;
    }
    .stDownloadButton > button:hover,
    .stButton > button[kind="primary"]:hover {
        background: #4F46E5;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6366F1, #8B5CF6);
        border-radius: 999px;
    }

    /* ── Slider accent ── */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #6366F1;
    }

    /* ── Semantic colors ── */
    .metric-positive { color: #22C55E !important; font-weight: 600; }
    .metric-negative { color: #EF4444 !important; font-weight: 600; }
    .metric-neutral  { color: #F59E0B !important; font-weight: 600; }
    </style>
    """
