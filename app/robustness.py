"""
robustness.py — Parameter sensitivity and stress tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def slippage_stress_test(tdf: pd.DataFrame, cfg: dict, slippage_range=None) -> pd.DataFrame:
    """Re-compute net P&L under different slippage assumptions."""
    if tdf.empty:
        return pd.DataFrame()
    if slippage_range is None:
        slippage_range = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]

    LOT_SIZE = cfg['LOT_SIZE']
    BROKERAGE = cfg['BROKERAGE_PER_TRADE']
    results = []

    for slip_rate in slippage_range:
        new_slip = tdf['gross_pnl'].abs() * slip_rate
        new_net = tdf['gross_pnl'] - new_slip - BROKERAGE
        total = new_net.sum()
        wr = (new_net > 0).mean() * 100
        results.append({
            'slippage_pct': f'{slip_rate * 100:.2f}%',
            'total_net_pnl': round(total, 2),
            'win_rate': round(wr, 1),
            'avg_trade_pnl': round(new_net.mean(), 2),
        })
    return pd.DataFrame(results)


def brokerage_stress_test(tdf: pd.DataFrame, cfg: dict, brokerage_range=None) -> pd.DataFrame:
    """Re-compute net P&L under different brokerage assumptions."""
    if tdf.empty:
        return pd.DataFrame()
    if brokerage_range is None:
        brokerage_range = [0, 20, 40, 60, 80, 100, 150, 200]

    SLIPPAGE_RATE = cfg['SLIPPAGE_RATE']
    results = []

    for brok in brokerage_range:
        slip = tdf['gross_pnl'].abs() * SLIPPAGE_RATE
        new_net = tdf['gross_pnl'] - slip - brok
        total = new_net.sum()
        wr = (new_net > 0).mean() * 100
        results.append({
            'brokerage_flat': f'Rs {brok}',
            'total_net_pnl': round(total, 2),
            'win_rate': round(wr, 1),
            'avg_trade_pnl': round(new_net.mean(), 2),
        })
    return pd.DataFrame(results)


def compute_rolling_sharpe(tdf: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Compute rolling Sharpe ratio over daily P&L."""
    if tdf.empty:
        return pd.DataFrame()
    daily = tdf.groupby('entry_date')['net_pnl'].sum().reset_index()
    daily.columns = ['date', 'net_pnl']
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)

    roll_mean = daily['net_pnl'].rolling(window, min_periods=max(5, window // 3)).mean()
    roll_std = daily['net_pnl'].rolling(window, min_periods=max(5, window // 3)).std()
    daily['rolling_sharpe'] = (roll_mean / roll_std * np.sqrt(252)).replace([np.inf, -np.inf], np.nan)

    return daily[['date', 'net_pnl', 'rolling_sharpe']].dropna()


def parameter_sensitivity_heatmap(tdf: pd.DataFrame, cfg: dict,
                                   stop_range=None, signal_range=None) -> pd.DataFrame:
    """Generate a stop-loss vs signal threshold sensitivity heatmap."""
    if tdf.empty:
        return pd.DataFrame()
    if stop_range is None:
        stop_range = [-0.0005, -0.001, -0.0015, -0.002, -0.0025, -0.003]
    if signal_range is None:
        signal_range = [0.25, 0.30, 0.32, 0.35, 0.40, 0.45]

    results = {}
    for sig in signal_range:
        row = {}
        for stop in stop_range:
            # Approximate: filter trades by signal threshold and adjust by stop ratio
            filtered = tdf[tdf['signal'] >= sig]
            if len(filtered) > 0:
                approx_net = filtered['net_pnl'].sum()
                stop_scale = stop / cfg['STOP_LOSS_PCT']
                loss_trades = filtered[filtered['net_pnl'] < 0]
                win_trades = filtered[filtered['net_pnl'] >= 0]
                approx_net = win_trades['net_pnl'].sum() + loss_trades['net_pnl'].sum() * stop_scale
            else:
                approx_net = 0
            row[f'{stop * 100:.2f}%'] = round(approx_net, 0)
        results[f'{sig:.2f}'] = row

    return pd.DataFrame(results).T
