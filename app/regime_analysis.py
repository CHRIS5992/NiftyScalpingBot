"""
regime_analysis.py — Market regime classification and performance breakdown.
Uses the notebook's 4-regime system (Trend HV, Trend LV, Range HV, Range LV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_MAP = {0: 'Trend HV', 1: 'Trend LV', 2: 'Range HV', 3: 'Range LV'}


def get_regime_distribution(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Regime distribution from feature-engineered data."""
    if df_feat.empty or 'market_regime' not in df_feat.columns:
        return pd.DataFrame()
    counts = df_feat['market_regime'].value_counts().sort_index()
    result = pd.DataFrame({
        'regime': [REGIME_MAP.get(i, f'R{i}') for i in counts.index],
        'count': counts.values,
        'pct': (counts.values / counts.values.sum() * 100).round(1),
    })
    return result


def get_regime_trade_performance(tdf: pd.DataFrame) -> pd.DataFrame:
    """Performance breakdown by the 4 market regimes from trades."""
    if tdf.empty or 'regime' not in tdf.columns:
        return pd.DataFrame()
    return (
        tdf.groupby('regime')
        .agg(
            trades=('net_pnl', 'count'),
            total_pnl=('net_pnl', 'sum'),
            avg_pnl=('net_pnl', 'mean'),
            win_rate=('net_pnl', lambda x: (x > 0).mean() * 100),
            avg_bars_held=('bars_held', 'mean'),
            avg_pnl_pct=('pnl_pct', 'mean'),
        )
        .round(2)
        .sort_values('total_pnl', ascending=False)
    )


def classify_vol_trend_regimes(df_raw: pd.DataFrame, tdf: pd.DataFrame) -> pd.DataFrame:
    """Classify additional volatility/trend regimes from raw data for visualization.

    Uses daily spot data to determine:
      - Trend regime: bull / bear / sideways (20-day SMA slope)
      - Volatility regime: high / low / normal (rolling vol percentile)
    """
    if tdf.empty or df_raw.empty:
        return tdf

    # Build daily summary
    daily = df_raw.groupby('date').agg(
        close_open=('close', 'first'),
        close_last=('close', 'last'),
    ).reset_index()

    # Trend classification using 20-day SMA
    daily['sma20'] = daily['close_last'].rolling(20, min_periods=5).mean()
    daily['sma_slope'] = daily['sma20'].pct_change(5)

    def _trend(slope):
        if pd.isna(slope):
            return 'Sideways'
        if slope > 0.005:
            return 'Bull'
        elif slope < -0.005:
            return 'Bear'
        return 'Sideways'

    daily['trend_regime'] = daily['sma_slope'].apply(_trend)

    # Volatility classification using price vol
    daily['rv'] = daily['close_last'].pct_change().rolling(20, min_periods=5).std()
    daily['rv_pct'] = daily['rv'].rolling(60, min_periods=10).rank(pct=True)

    def _vol(pct):
        if pd.isna(pct):
            return 'Normal'
        if pct > 0.7:
            return 'High Volatility'
        elif pct < 0.3:
            return 'Low Volatility'
        return 'Normal'

    daily['vol_regime_label'] = daily['rv_pct'].apply(_vol)

    regime_map = daily.set_index('date')[['trend_regime', 'vol_regime_label']]
    tdf_out = tdf.copy()
    tdf_out['trend_regime'] = tdf_out['entry_date'].map(regime_map['trend_regime']).fillna('Sideways')
    tdf_out['vol_regime_label'] = tdf_out['entry_date'].map(regime_map['vol_regime_label']).fillna('Normal')

    return tdf_out


def get_regime_performance(tdf: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    """Performance breakdown by any regime column."""
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
