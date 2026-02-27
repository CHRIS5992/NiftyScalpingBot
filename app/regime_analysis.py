"""
regime_analysis.py — Market regime classification and performance breakdown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def classify_regimes(df_raw: pd.DataFrame, tdf: pd.DataFrame) -> pd.DataFrame:
    """Classify market regimes and compute performance per regime.

    Uses daily spot data to determine:
      - Trend regime: bull / bear / sideways (based on 20-day SMA slope)
      - Volatility regime: high / low (based on 20-day IV percentile)

    Returns tdf with regime columns attached.
    """
    if tdf.empty or df_raw.empty:
        return tdf

    # Build daily summary from raw data
    daily = df_raw.groupby('date').agg(
        spot_open=('spot', 'first'),
        spot_close=('spot', 'last'),
        avg_iv=('iv', 'mean'),
    ).reset_index()
    daily['date'] = daily['date'].astype(str)

    # Trend classification using 20-day SMA
    daily['sma20'] = daily['spot_close'].rolling(20, min_periods=5).mean()
    daily['sma_slope'] = daily['sma20'].pct_change(5)  # 5-day slope

    def _trend(slope):
        if pd.isna(slope):
            return 'Sideways'
        if slope > 0.005:
            return 'Bull'
        elif slope < -0.005:
            return 'Bear'
        return 'Sideways'

    daily['trend_regime'] = daily['sma_slope'].apply(_trend)

    # Volatility classification using IV percentile
    daily['iv_pct'] = daily['avg_iv'].rolling(60, min_periods=10).rank(pct=True)

    def _vol(pct):
        if pd.isna(pct):
            return 'Normal'
        if pct > 0.7:
            return 'High Volatility'
        elif pct < 0.3:
            return 'Low Volatility'
        return 'Normal'

    daily['vol_regime'] = daily['iv_pct'].apply(_vol)

    # Merge into trade df
    regime_map = daily.set_index('date')[['trend_regime', 'vol_regime']]
    tdf_out = tdf.copy()
    tdf_out['trend_regime'] = tdf_out['date'].map(regime_map['trend_regime']).fillna('Sideways')
    tdf_out['vol_regime']   = tdf_out['date'].map(regime_map['vol_regime']).fillna('Normal')

    return tdf_out


def get_regime_performance(tdf: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    """Performance breakdown by a regime column."""
    if tdf.empty or regime_col not in tdf.columns:
        return pd.DataFrame()
    return (
        tdf.groupby(regime_col)
        .agg(
            trades=('net_pnl', 'count'),
            total_pnl=('net_pnl', 'sum'),
            avg_pnl=('net_pnl', 'mean'),
            win_rate=('net_pnl', lambda x: (x > 0).mean() * 100),
            avg_bars_held=('bars_held', 'mean'),
        )
        .round(2)
        .sort_values('total_pnl', ascending=False)
    )
