"""
metrics.py — P&L analytics and performance metrics.
Matched to Final_straddle.ipynb outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _max_streak(series, positive: bool = True) -> int:
    """Return longest consecutive streak of positive (or negative) values."""
    st, mx = 0, 0
    for v in series:
        if (v > 0) == positive:
            st += 1
            mx = max(mx, st)
        else:
            st = 0
    return mx


def compute_metrics(tdf: pd.DataFrame, cfg: dict) -> dict:
    """Compute all performance metrics matching notebook Cell 8 output."""
    if tdf.empty:
        return _empty_metrics()

    n = len(tdf)
    wins = tdf[tdf['net_pnl'] > 0]
    losses = tdf[tdf['net_pnl'] <= 0]
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n * 100

    avg_win = wins['net_pnl'].mean() if n_wins > 0 else 0
    avg_loss = losses['net_pnl'].mean() if n_losses > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    total_gross = tdf['gross_pnl'].sum()
    total_slippage = tdf['slippage'].sum()
    total_brokerage = cfg['BROKERAGE_PER_TRADE'] * n
    total_net = tdf['net_pnl'].sum()
    roi = total_net / cfg['INITIAL_CAPITAL'] * 100

    # Profit factor — exact notebook formula
    gross_wins = wins['net_pnl'].sum() if n_wins > 0 else 0
    gross_losses_abs = abs(losses['net_pnl'].sum()) if n_losses > 0 else 1e-8
    profit_factor = gross_wins / gross_losses_abs if gross_losses_abs > 0 else float('inf')

    # Cumulative & drawdown
    cum = tdf['net_pnl'].cumsum()
    peak = cum.cummax()
    dd_ser = cum - peak
    max_dd = dd_ser.min()
    max_dd_pct = ((cum - peak) / (cfg['INITIAL_CAPITAL'] + peak)).min() * 100

    # Daily P&L
    daily_p = tdf.groupby('entry_date')['net_pnl'].sum()
    sharpe = (daily_p.mean() / daily_p.std() * np.sqrt(252)
              if daily_p.std() > 0 else 0)

    neg_ret = daily_p[daily_p < 0]
    downside_std = neg_ret.std() if len(neg_ret) > 1 else 1e-8
    sortino = daily_p.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

    # CAGR
    start_dt = pd.to_datetime(tdf['entry_date'].min())
    end_dt = pd.to_datetime(tdf['exit_date'].max())
    years = max((end_dt - start_dt).days / 365.25, 0.01)
    final_capital = cfg['INITIAL_CAPITAL'] + total_net
    cagr = ((final_capital / cfg['INITIAL_CAPITAL']) ** (1 / years) - 1) * 100

    avg_hold = tdf['bars_held'].mean()
    max_hold = tdf['bars_held'].max()

    return {
        'total_trades': n,
        'trading_days': tdf['entry_date'].nunique(),
        'winning_trades': n_wins,
        'losing_trades': n_losses,
        'win_rate': round(win_rate, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'avg_win_pct': round(wins['pnl_pct'].mean(), 4) if n_wins > 0 else 0,
        'avg_loss_pct': round(losses['pnl_pct'].mean(), 4) if n_losses > 0 else 0,
        'risk_reward': round(rr, 2),
        'profit_factor': round(profit_factor, 2),
        'gross_pnl': round(total_gross, 2),
        'slippage_total': round(total_slippage, 2),
        'brokerage_total': round(total_brokerage, 2),
        'total_costs': round(total_slippage + total_brokerage, 2),
        'net_pnl': round(total_net, 2),
        'roi': round(roi, 2),
        'cagr': round(cagr, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_drawdown': round(max_dd, 2),
        'max_drawdown_pct': round(max_dd_pct, 2),
        'avg_hold_bars': round(avg_hold, 1),
        'max_hold_bars': int(max_hold),
        'avg_hold_min': round(avg_hold * 5, 1),
        'max_hold_min': int(max_hold * 5),
        'max_con_wins': _max_streak(tdf['net_pnl'], True),
        'max_con_losses': _max_streak(tdf['net_pnl'], False),
        'final_capital': round(final_capital, 2),
        'cum_pnl': cum,
        'dd_series': dd_ser,
    }


def _empty_metrics() -> dict:
    return {k: 0 for k in [
        'total_trades', 'trading_days', 'winning_trades', 'losing_trades',
        'win_rate', 'avg_win', 'avg_loss', 'avg_win_pct', 'avg_loss_pct',
        'risk_reward', 'profit_factor', 'gross_pnl', 'slippage_total',
        'brokerage_total', 'total_costs', 'net_pnl', 'roi', 'cagr',
        'sharpe', 'sortino', 'max_drawdown', 'max_drawdown_pct',
        'avg_hold_bars', 'max_hold_bars', 'avg_hold_min', 'max_hold_min',
        'max_con_wins', 'max_con_losses', 'final_capital',
    ]}


def compute_monthly_breakdown(tdf: pd.DataFrame) -> pd.DataFrame:
    """Monthly P&L breakdown."""
    if tdf.empty:
        return pd.DataFrame()
    mp = tdf.copy()
    mp['month'] = pd.to_datetime(mp['entry_date']).dt.to_period('M')
    return (
        mp.groupby('month')
        .agg(
            trades=('net_pnl', 'count'),
            net_pnl=('net_pnl', 'sum'),
            win_rate=('net_pnl', lambda x: (x > 0).mean() * 100),
        )
        .round(2)
    )


def compute_monthly_heatmap_data(tdf: pd.DataFrame) -> pd.DataFrame:
    """Pivoted monthly data for heatmap — matches notebook Cell 11."""
    if tdf.empty:
        return pd.DataFrame()
    mp = tdf.copy()
    mp['entry_date'] = pd.to_datetime(mp['entry_date'])
    mp['Year'] = mp['entry_date'].dt.year
    mp['Month'] = mp['entry_date'].dt.month_name().str[:3]
    heatmap_data = mp.pivot_table(
        values='net_pnl', index='Year', columns='Month', aggfunc='sum', fill_value=0
    )
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    heatmap_data = heatmap_data.reindex(
        columns=[m for m in months_order if m in heatmap_data.columns]
    )
    heatmap_data.index = [str(y) for y in heatmap_data.index]
    return heatmap_data


def compute_exit_breakdown(tdf: pd.DataFrame) -> pd.DataFrame:
    if tdf.empty:
        return pd.DataFrame()
    vc = tdf['exit_reason'].value_counts()
    result = pd.DataFrame({'exit_reason': vc.index, 'count': vc.values})
    result['pct'] = (result['count'] / result['count'].sum() * 100).round(1)
    # Add avg P&L per exit reason
    avg_pnl = tdf.groupby('exit_reason')['net_pnl'].mean()
    result['avg_pnl'] = result['exit_reason'].map(avg_pnl).round(2)
    return result


def compute_lot_performance(tdf: pd.DataFrame) -> pd.DataFrame:
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


def compute_regime_performance(tdf: pd.DataFrame, regime_map: dict) -> pd.DataFrame:
    """Regime performance — matches notebook Cell 8."""
    if tdf.empty:
        return pd.DataFrame()
    perf = tdf.groupby('regime').agg(
        trades=('net_pnl', 'count'),
        total_pnl=('net_pnl', 'sum'),
        avg_pnl=('net_pnl', 'mean'),
        win_rate=('net_pnl', lambda x: (x > 0).mean() * 100),
        avg_pnl_pct=('pnl_pct', 'mean'),
    ).round(2)
    perf.index = perf.index.map(lambda x: regime_map.get(x, f'Regime {x}'))
    return perf


def compute_entry_quality(tdf: pd.DataFrame) -> dict:
    """Entry quality analysis."""
    if tdf.empty:
        return {}
    winners = tdf[tdf['net_pnl'] > 0]
    losers = tdf[tdf['net_pnl'] <= 0]
    return {
        'winners_signal': round(winners['signal'].mean(), 4) if len(winners) > 0 else 0,
        'losers_signal': round(losers['signal'].mean(), 4) if len(losers) > 0 else 0,
        'winners_bars_held': round(winners['bars_held'].mean(), 1) if len(winners) > 0 else 0,
        'losers_bars_held': round(losers['bars_held'].mean(), 1) if len(losers) > 0 else 0,
        'winners_avg_regime': round(winners['regime'].mean(), 2) if len(winners) > 0 else 0,
        'losers_avg_regime': round(losers['regime'].mean(), 2) if len(losers) > 0 else 0,
    }


def compute_weekday_pnl(tdf: pd.DataFrame) -> pd.DataFrame:
    if tdf.empty:
        return pd.DataFrame()
    tmp = tdf.copy()
    tmp['weekday'] = pd.to_datetime(tmp['entry_date']).dt.day_name()
    return tmp.groupby('weekday').agg(
        total_pnl=('net_pnl', 'sum'),
        count=('net_pnl', 'count'),
        avg_pnl=('net_pnl', 'mean'),
        win_rate=('net_pnl', lambda x: (x > 0).mean() * 100),
    ).round(2)


def compute_hourly_pnl(tdf: pd.DataFrame) -> pd.DataFrame:
    """P&L by hour of day — matches notebook Cell 11 chart 9."""
    if tdf.empty:
        return pd.DataFrame()
    tmp = tdf.copy()
    tmp['hour'] = pd.to_datetime(tmp['entry_time']).dt.hour
    return tmp.groupby('hour').agg(
        total_pnl=('net_pnl', 'sum'),
        count=('net_pnl', 'count'),
        avg_pnl=('net_pnl', 'mean'),
    ).round(2)
