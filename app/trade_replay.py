"""
trade_replay.py — Intraday day replay logic.
"""
from __future__ import annotations

import pandas as pd


def get_day_data(
    df_raw: pd.DataFrame,
    tdf: pd.DataFrame,
    selected_date: str,
) -> dict:
    """Get all data needed to replay a single trading day.

    Args:
        df_raw: Raw cleaned data with all bars.
        tdf: Trades DataFrame.
        selected_date: Date string 'YYYY-MM-DD'.

    Returns:
        Dict with day_bars, day_trades, summary.
    """
    day_bars = df_raw[df_raw['date'] == selected_date].copy()
    day_trades = tdf[tdf['date'] == selected_date].copy() if not tdf.empty else pd.DataFrame()

    if day_bars.empty:
        return {
            'day_bars': day_bars,
            'day_trades': day_trades,
            'summary': {},
        }

    summary = {
        'date': selected_date,
        'spot_open': round(day_bars['spot'].iloc[0], 2),
        'spot_close': round(day_bars['spot'].iloc[-1], 2),
        'spot_high': round(day_bars['spot'].max(), 2),
        'spot_low': round(day_bars['spot'].min(), 2),
        'day_range': round(day_bars['spot'].max() - day_bars['spot'].min(), 2),
        'straddle_open': round(day_bars['Straddle_Price'].iloc[0], 2),
        'straddle_close': round(day_bars['Straddle_Price'].iloc[-1], 2),
        'avg_iv': round(day_bars['iv'].mean(), 2),
        'num_bars': len(day_bars),
        'num_trades': len(day_trades),
        'day_pnl': round(day_trades['net_pnl'].sum(), 2) if not day_trades.empty else 0,
    }

    return {
        'day_bars': day_bars,
        'day_trades': day_trades,
        'summary': summary,
    }


def get_available_dates(tdf: pd.DataFrame) -> list[str]:
    """Return sorted list of dates that have trades."""
    if tdf.empty:
        return []
    return sorted(tdf['date'].unique().tolist())
