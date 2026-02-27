"""
robustness.py — Parameter sensitivity and stress tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def slippage_stress_test(tdf: pd.DataFrame, cfg: dict, slippage_range: list[float] | None = None) -> pd.DataFrame:
    """Re-compute net P&L under different slippage assumptions.

    Rather than re-running the full backtest (which would change entry decisions),
    we adjust costs on the existing trade log for speed.
    """
    if tdf.empty:
        return pd.DataFrame()

    if slippage_range is None:
        slippage_range = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]

    LOT_SIZE = cfg['LOT_SIZE']
    BROKERAGE_FLAT = cfg['BROKERAGE_FLAT']
    results = []

    for slip_pct in slippage_range:
        new_slip = (tdf['entry_price'] + tdf['exit_price']) * slip_pct * LOT_SIZE * tdf['lots']
        new_net  = tdf['gross_pnl'] - new_slip - BROKERAGE_FLAT
        total    = new_net.sum()
        wr       = (new_net > 0).mean() * 100
        results.append({
            'slippage_pct': f'{slip_pct*100:.2f}%',
            'total_net_pnl': round(total, 2),
            'win_rate': round(wr, 1),
            'avg_trade_pnl': round(new_net.mean(), 2),
        })

    return pd.DataFrame(results)


def brokerage_stress_test(tdf: pd.DataFrame, cfg: dict, brokerage_range: list[float] | None = None) -> pd.DataFrame:
    """Re-compute net P&L under different brokerage assumptions."""
    if tdf.empty:
        return pd.DataFrame()

    if brokerage_range is None:
        brokerage_range = [0, 20, 40, 60, 80, 100, 150, 200]

    SLIPPAGE_PCT = cfg['SLIPPAGE_PCT']
    LOT_SIZE = cfg['LOT_SIZE']
    results = []

    for brok in brokerage_range:
        slip = (tdf['entry_price'] + tdf['exit_price']) * SLIPPAGE_PCT * LOT_SIZE * tdf['lots']
        new_net = tdf['gross_pnl'] - slip - brok
        total = new_net.sum()
        wr = (new_net > 0).mean() * 100
        results.append({
            'brokerage_flat': f'₹{brok}',
            'total_net_pnl': round(total, 2),
            'win_rate': round(wr, 1),
            'avg_trade_pnl': round(new_net.mean(), 2),
        })

    return pd.DataFrame(results)


def compute_rolling_sharpe(tdf: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Compute rolling Sharpe ratio over daily P&L.

    Args:
        tdf: Trades DataFrame.
        window: Rolling window in trading days.
    """
    if tdf.empty:
        return pd.DataFrame()

    daily = tdf.groupby('date')['net_pnl'].sum().reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)

    roll_mean = daily['net_pnl'].rolling(window, min_periods=max(5, window // 3)).mean()
    roll_std  = daily['net_pnl'].rolling(window, min_periods=max(5, window // 3)).std()
    daily['rolling_sharpe'] = (roll_mean / roll_std * np.sqrt(252)).replace([np.inf, -np.inf], np.nan)

    return daily[['date', 'net_pnl', 'rolling_sharpe']].dropna()


def parameter_sensitivity_heatmap(
    tdf: pd.DataFrame,
    cfg: dict,
    stop_range: list[float] | None = None,
    target_range: list[float] | None = None,
) -> pd.DataFrame:
    """Generate a stop-loss vs target sensitivity heatmap.

    Uses existing trade data and adjusts exit thresholds (approximate).
    For exact results this would require re-running backtest, but for
    visualization purposes we compute the net impact on existing trades.
    """
    if tdf.empty:
        return pd.DataFrame()

    if stop_range is None:
        stop_range = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    if target_range is None:
        target_range = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

    # For each stop/target combo, count how many of existing trades would
    # have been profitable vs losing based on their pnl_pct
    results = {}
    for target in target_range:
        row = {}
        for stop in stop_range:
            # Approximate: trades with pnl_pct > target*100 -> win, < -stop*100 -> loss
            wins_pnl = tdf[tdf['pnl_pct'] > 0]['net_pnl'].sum()
            losses_pnl = tdf[tdf['pnl_pct'] <= 0]['net_pnl'].sum()
            # Scale by ratio of original parameters
            target_scale = target / cfg['TARGET_PCT']
            stop_scale = stop / cfg['STOP_PCT']
            approx_net = wins_pnl * target_scale + losses_pnl * stop_scale
            row[f'{stop*100:.0f}%'] = round(approx_net, 0)
        results[f'{target*100:.0f}%'] = row

    return pd.DataFrame(results).T
