"""
iv_analytics.py — IV/Volatility analytics module.
Adapted for the regime-switch bot (uses hv_short/hv_long instead of options IV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vol_percentile_series(df_feat: pd.DataFrame, window: int = 500) -> pd.DataFrame:
    """Compute rolling volatility percentile over time."""
    if df_feat.empty or 'hv_short' not in df_feat.columns:
        return pd.DataFrame()
    daily = df_feat.groupby('date').agg(
        avg_hv=('hv_short', 'mean'),
    ).reset_index()
    daily['vol_pct'] = daily['avg_hv'].rolling(min(window, len(daily)), min_periods=20).rank(pct=True) * 100
    daily['date'] = pd.to_datetime(daily['date'])
    return daily


def compute_vol_vs_returns(tdf: pd.DataFrame) -> dict:
    """Volatility regime vs returns analysis."""
    if tdf.empty:
        return {}
    from .config import REGIME_MAP
    regime_perf = tdf.groupby('regime')['net_pnl'].agg(['sum', 'mean', 'count'])
    return {
        'regime_perf': regime_perf,
    }


def compute_vol_change_analysis(tdf: pd.DataFrame) -> dict:
    """Analyze signal vs P&L relationship."""
    if tdf.empty:
        return {}
    winners = tdf[tdf['net_pnl'] > 0]
    losers = tdf[tdf['net_pnl'] <= 0]
    return {
        'avg_signal': round(tdf['signal'].mean(), 4),
        'winners_avg_signal': round(winners['signal'].mean(), 4) if len(winners) > 0 else 0,
        'losers_avg_signal': round(losers['signal'].mean(), 4) if len(losers) > 0 else 0,
        'signal_pnl_corr': round(tdf['signal'].corr(tdf['net_pnl']), 3),
    }
