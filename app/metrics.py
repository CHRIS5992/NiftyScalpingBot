"""
metrics.py — P&L analytics and performance metrics.
Exact reproduction of notebook Cells 7, 8.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import max_streak


def compute_metrics(tdf: pd.DataFrame, cfg: dict) -> dict:
    """Compute all performance metrics. Exact Cell 7 logic.

    Returns a dict with all KPIs matching notebook output.
    """
    if tdf.empty:
        return _empty_metrics()

    n = len(tdf)
    wins = tdf[tdf['net_pnl'] > 0]
    loss = tdf[tdf['net_pnl'] <= 0]
    wc = len(wins)
    lc = len(loss)
    wr = wc / n * 100

    avg_w = wins['net_pnl'].mean() if wc > 0 else 0
    avg_l = loss['net_pnl'].mean() if lc > 0 else 0
    rr    = abs(avg_w / avg_l) if avg_l != 0 else 0

    gross_pnl      = tdf['gross_pnl'].sum()
    slippage_total  = tdf['slip_cost'].sum()
    brokerage_total = cfg['BROKERAGE_FLAT'] * n
    taxes           = gross_pnl * cfg.get('TAX_RATE', 0.15)
    costs           = slippage_total + brokerage_total + taxes
    net_pnl         = tdf['net_pnl'].sum()
    roi             = net_pnl / cfg['INITIAL_CAPITAL'] * 100

    cum     = tdf['net_pnl'].cumsum()
    dd_ser  = cum - cum.cummax()
    max_dd  = dd_ser.min()

    daily_p = tdf.groupby('date')['net_pnl'].sum()
    sharpe  = (daily_p.mean() / daily_p.std() * np.sqrt(252)
               if daily_p.std() > 0 else 0)

    # Sortino ratio
    neg_ret = daily_p[daily_p < 0]
    downside_std = neg_ret.std() if len(neg_ret) > 1 else 1e-8
    sortino = daily_p.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

    # Profit factor
    gross_wins  = wins['net_pnl'].sum() if wc > 0 else 0
    gross_losses = abs(loss['net_pnl'].sum()) if lc > 0 else 1e-8
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0

    # CAGR
    start_dt = pd.to_datetime(tdf['date'].min())
    end_dt   = pd.to_datetime(tdf['date'].max())
    years    = max((end_dt - start_dt).days / 365.25, 0.01)
    final_capital = cfg['INITIAL_CAPITAL'] + net_pnl
    cagr = ((final_capital / cfg['INITIAL_CAPITAL']) ** (1 / years) - 1) * 100

    # Avg/Max holding time
    avg_hold_bars = tdf['bars_held'].mean()
    max_hold_bars = tdf['bars_held'].max()
    timeframe_min = 5 if cfg.get('USE_TIMEFRAME', '5min') == '5min' else 1
    avg_hold_min  = avg_hold_bars * timeframe_min
    max_hold_min  = max_hold_bars * timeframe_min

    return {
        'total_trades':     n,
        'trading_days':     tdf['date'].nunique(),
        'avg_trades_day':   round(n / max(tdf['date'].nunique(), 1), 1),
        'winning_trades':   wc,
        'losing_trades':    lc,
        'win_rate':         round(wr, 1),
        'avg_win':          round(avg_w, 2),
        'avg_loss':         round(avg_l, 2),
        'avg_win_pct':      round(wins['pnl_pct'].mean(), 2) if wc > 0 else 0,
        'avg_loss_pct':     round(loss['pnl_pct'].mean(), 2) if lc > 0 else 0,
        'risk_reward':      round(rr, 2),
        'profit_factor':    round(profit_factor, 2),
        'gross_pnl':        round(gross_pnl, 2),
        'slippage_total':   round(slippage_total, 2),
        'brokerage_total':  round(brokerage_total, 2),
        'taxes':            round(taxes, 2),
        'total_costs':      round(costs, 2),
        'net_pnl':          round(net_pnl, 2),
        'roi':              round(roi, 2),
        'cagr':             round(cagr, 2),
        'total_return_pct': round(roi, 2),
        'sharpe':           round(sharpe, 2),
        'sortino':          round(sortino, 2),
        'max_drawdown':     round(max_dd, 2),
        'avg_hold_bars':    round(avg_hold_bars, 1),
        'max_hold_bars':    int(max_hold_bars),
        'avg_hold_min':     round(avg_hold_min, 1),
        'max_hold_min':     int(max_hold_min),
        'max_con_wins':     max_streak(tdf['net_pnl'], True),
        'max_con_losses':   max_streak(tdf['net_pnl'], False),
        'period_start':     tdf['date'].min(),
        'period_end':       tdf['date'].max(),
        'cum_pnl':          cum,
        'dd_series':        dd_ser,
        'final_capital':    round(final_capital, 2),
    }


def compute_monthly_breakdown(tdf: pd.DataFrame) -> pd.DataFrame:
    """Return monthly P&L breakdown. Exact Cell 7 logic."""
    if tdf.empty:
        return pd.DataFrame()
    mp = tdf.copy()
    mp['month'] = pd.to_datetime(mp['date']).dt.to_period('M')
    monthly = (
        mp.groupby('month')
        .agg(
            trades=('net_pnl', 'count'),
            net_pnl=('net_pnl', 'sum'),
            win_rate=('net_pnl', lambda x: (x > 0).mean() * 100),
        )
        .round(2)
    )
    return monthly


def compute_monthly_heatmap_data(tdf: pd.DataFrame) -> pd.DataFrame:
    """Return pivoted monthly data for heatmap. Exact Cell 12 logic."""
    if tdf.empty:
        return pd.DataFrame()
    mp = tdf.copy()
    mp['yr'] = pd.to_datetime(mp['date']).dt.year
    mp['mo'] = pd.to_datetime(mp['date']).dt.month
    piv = mp.groupby(['yr', 'mo'])['net_pnl'].sum().unstack(fill_value=0)
    mlbls = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    piv.columns = [mlbls[c - 1] for c in piv.columns]
    piv.index = [str(y) for y in piv.index]
    return piv


def compute_exit_breakdown(tdf: pd.DataFrame) -> pd.DataFrame:
    """Return exit reason value counts."""
    if tdf.empty:
        return pd.DataFrame()
    return tdf['exit_reason'].value_counts().reset_index()


def compute_lot_performance(tdf: pd.DataFrame) -> pd.DataFrame:
    """Return performance breakdown by lot size. Exact Cell 7 logic."""
    if tdf.empty:
        return pd.DataFrame()
    return (
        tdf.groupby('lots')
        .agg(
            count=('net_pnl', 'count'),
            total_pnl=('net_pnl', 'sum'),
            avg_pnl=('net_pnl', 'mean'),
            win_rate=('net_pnl', lambda x: (x > 0).mean() * 100),
        )
        .round(2)
    )


def compute_entry_quality(tdf: pd.DataFrame) -> dict:
    """Entry quality analysis. Exact Cell 8 logic."""
    if tdf.empty:
        return {}
    winners = tdf[tdf['net_pnl'] > 0]
    losers  = tdf[tdf['net_pnl'] <= 0]
    return {
        'winners_ml_signal':  round(winners['ml_signal'].mean(), 4) if len(winners) > 0 else 0,
        'losers_ml_signal':   round(losers['ml_signal'].mean(), 4) if len(losers) > 0 else 0,
        'winners_entry_iv':   round(winners['entry_iv'].mean(), 2) if len(winners) > 0 else 0,
        'losers_entry_iv':    round(losers['entry_iv'].mean(), 2) if len(losers) > 0 else 0,
        'winners_bars_held':  round(winners['bars_held'].mean(), 1) if len(winners) > 0 else 0,
        'losers_bars_held':   round(losers['bars_held'].mean(), 1) if len(losers) > 0 else 0,
    }


def compute_daily_pnl(tdf: pd.DataFrame) -> pd.DataFrame:
    """Daily P&L aggregation."""
    if tdf.empty:
        return pd.DataFrame()
    return tdf.groupby('date')['net_pnl'].sum().reset_index()


def compute_weekday_pnl(tdf: pd.DataFrame) -> pd.DataFrame:
    """P&L breakdown by day of week."""
    if tdf.empty:
        return pd.DataFrame()
    tmp = tdf.copy()
    tmp['weekday'] = pd.to_datetime(tmp['date']).dt.day_name()
    return tmp.groupby('weekday').agg(
        total_pnl=('net_pnl', 'sum'),
        count=('net_pnl', 'count'),
        avg_pnl=('net_pnl', 'mean'),
        win_rate=('net_pnl', lambda x: (x > 0).mean() * 100),
    ).round(2)


def _empty_metrics() -> dict:
    """Return zero-filled metrics dict when no trades."""
    return {k: 0 for k in [
        'total_trades', 'trading_days', 'avg_trades_day',
        'winning_trades', 'losing_trades', 'win_rate',
        'avg_win', 'avg_loss', 'avg_win_pct', 'avg_loss_pct',
        'risk_reward', 'profit_factor', 'gross_pnl',
        'slippage_total', 'brokerage_total', 'taxes', 'total_costs',
        'net_pnl', 'roi', 'cagr', 'total_return_pct',
        'sharpe', 'sortino', 'max_drawdown',
        'avg_hold_bars', 'max_hold_bars', 'avg_hold_min', 'max_hold_min',
        'max_con_wins', 'max_con_losses',
        'period_start', 'period_end', 'final_capital',
    ]}
