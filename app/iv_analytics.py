"""
iv_analytics.py — IV analytics module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_iv_percentile_series(df_raw: pd.DataFrame, window: int = 78) -> pd.DataFrame:
    """Compute rolling IV percentile over time.

    Uses the same ROLL_RANK window as the notebook.
    """
    daily = df_raw.groupby('date').agg(
        avg_iv=('iv', 'mean'),
        close_iv=('iv', 'last'),
    ).reset_index()
    daily['iv_pct'] = daily['close_iv'].rolling(window, min_periods=20).rank(pct=True) * 100
    daily['date'] = pd.to_datetime(daily['date'])
    return daily


def compute_iv_vs_returns(tdf: pd.DataFrame) -> dict:
    """IV vs returns analysis. Based on notebook Cell 13."""
    if tdf.empty:
        return {}
    corr = tdf['entry_iv'].corr(tdf['net_pnl'])
    z = np.polyfit(tdf['entry_iv'], tdf['net_pnl'], 1) if len(tdf) > 1 else [0, 0]
    return {
        'correlation': round(corr, 3),
        'slope': round(z[0], 2),
        'intercept': round(z[1], 2),
    }


def classify_iv_regime(tdf: pd.DataFrame, low_thresh: float = 12.0, high_thresh: float = 18.0) -> pd.DataFrame:
    """Classify each trade's IV regime.

    Args:
        tdf: Trades DataFrame.
        low_thresh: IV below this is 'Low IV'.
        high_thresh: IV above this is 'High IV'.
    """
    if tdf.empty:
        return tdf
    out = tdf.copy()
    conditions = [
        out['entry_iv'] < low_thresh,
        out['entry_iv'] > high_thresh,
    ]
    choices = ['Low IV', 'High IV']
    out['iv_regime'] = np.select(conditions, choices, default='Medium IV')
    return out


def get_iv_regime_performance(tdf: pd.DataFrame) -> pd.DataFrame:
    """Performance breakdown by IV regime."""
    if tdf.empty or 'iv_regime' not in tdf.columns:
        return pd.DataFrame()
    return (
        tdf.groupby('iv_regime')
        .agg(
            trades=('net_pnl', 'count'),
            total_pnl=('net_pnl', 'sum'),
            avg_pnl=('net_pnl', 'mean'),
            win_rate=('net_pnl', lambda x: (x > 0).mean() * 100),
            avg_iv=('entry_iv', 'mean'),
        )
        .round(2)
        .sort_values('avg_iv')
    )


def compute_iv_change_analysis(tdf: pd.DataFrame) -> dict:
    """IV change analysis from Cell 19/21."""
    if tdf.empty:
        return {}
    pos_iv  = tdf[tdf['iv_change'] > 0]
    neg_iv  = tdf[tdf['iv_change'] <= 0]
    winners = tdf[tdf['net_pnl'] > 0]
    losers  = tdf[tdf['net_pnl'] <= 0]
    return {
        'avg_iv_change':        round(tdf['iv_change'].mean(), 2),
        'std_iv_change':        round(tdf['iv_change'].std(), 2),
        'pos_iv_change_count':  len(pos_iv),
        'neg_iv_change_count':  len(neg_iv),
        'pos_iv_change_pct':    round(len(pos_iv) / len(tdf) * 100, 1),
        'winners_avg_iv_change': round(winners['iv_change'].mean(), 2) if len(winners) > 0 else 0,
        'losers_avg_iv_change':  round(losers['iv_change'].mean(), 2) if len(losers) > 0 else 0,
        'iv_pnl_corr':          round(tdf['iv_change'].corr(tdf['net_pnl']), 3),
        'ml_iv_corr':           round(tdf['ml_signal'].corr(tdf['iv_change']), 3),
    }
