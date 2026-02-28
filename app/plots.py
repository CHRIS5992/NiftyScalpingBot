"""
plots.py — All Plotly interactive charts for the Regime-Switch Bot.
Recreates every notebook chart in Plotly with dark theme.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Color palette
GOLD   = '#FFD700'
GREEN  = '#00E676'
RED    = '#FF5252'
DARK   = '#0d1117'
BLUE   = '#40C4FF'
ORANGE = '#FF9800'
PURPLE = '#CE93D8'

REGIME_COLORS = {0: '#d62728', 1: '#2ca02c', 2: '#ff7f0e', 3: '#1f77b4'}
REGIME_MAP = {0: 'Trend HV', 1: 'Trend LV', 2: 'Range HV', 3: 'Range LV'}

_LAYOUT = dict(
    paper_bgcolor=DARK,
    plot_bgcolor='#161b22',
    font=dict(family='Inter, sans-serif', color='#e6edf3', size=12),
    margin=dict(l=60, r=30, t=60, b=50),
    hovermode='x unified',
    legend=dict(bgcolor='#161b22', bordercolor='#21262d', borderwidth=1),
    xaxis=dict(gridcolor='#21262d', zerolinecolor='#21262d'),
    yaxis=dict(gridcolor='#21262d', zerolinecolor='#21262d'),
)


def _apply_layout(fig, title='', height=500, **kwargs):
    layout = {
        **_LAYOUT,
        'title': dict(text=title, font=dict(color=GOLD, size=16), x=0.5),
        'height': height,
    }
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


# ============================================================
# 1. Equity Curve (Cumulative P&L)
# ============================================================
def plot_equity_curve(tdf, cfg, metrics):
    """Cumulative equity curve — matches notebook chart 1."""
    if tdf.empty:
        return go.Figure()
    cum_pnl = tdf['net_pnl'].cumsum()
    capital = cfg['INITIAL_CAPITAL'] + cum_pnl

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(tdf) + 1)), y=cum_pnl,
        mode='lines', name='Cumulative P&L',
        line=dict(color=GREEN, width=3),
        fill='tozeroy', fillcolor='rgba(0,230,118,0.08)',
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3)

    net = metrics.get('net_pnl', 0)
    roi = metrics.get('roi', 0)
    return _apply_layout(fig, f'Cumulative Equity Curve  |  Net P&L: Rs {net:,.0f} (ROI: {roi:.2f}%)',
                         height=500, xaxis_title='Trade Number', yaxis_title='Net Profit (Rs)')


# ============================================================
# 2. Drawdown Chart
# ============================================================
def plot_drawdown(tdf, cfg):
    """Drawdown chart — matches notebook chart 2."""
    if tdf.empty:
        return go.Figure()
    cumulative = tdf['net_pnl'].cumsum()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / (cfg['INITIAL_CAPITAL'] + peak) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(tdf) + 1)), y=drawdown,
        mode='lines', name='Drawdown',
        line=dict(color=RED, width=2),
        fill='tozeroy', fillcolor='rgba(255,82,82,0.2)',
    ))
    return _apply_layout(fig, 'Historical Drawdown (%)', height=400,
                         xaxis_title='Trade Number', yaxis_title='Drawdown %')


# ============================================================
# 3. Monthly P&L Heatmap
# ============================================================
def plot_monthly_heatmap(piv):
    """Monthly heatmap — matches notebook chart 3."""
    if piv.empty:
        return go.Figure()
    max_val = np.abs(piv.values).max()
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
        colorscale=[[0, RED], [0.5, DARK], [1, GREEN]],
        zmin=-max_val, zmax=max_val,
        text=[[f'Rs {v:,.0f}' for v in row] for row in piv.values],
        texttemplate='%{text}', textfont=dict(size=11, color='white'),
        hovertemplate='Year: %{y}<br>Month: %{x}<br>P&L: %{text}<extra></extra>',
        colorbar=dict(title='P&L (Rs)'),
    ))
    return _apply_layout(fig, 'Monthly Returns Heatmap', height=max(350, len(piv) * 80))


# ============================================================
# 4. Rolling Win Rate
# ============================================================
def plot_rolling_win_rate(tdf, window=50):
    """Rolling win rate — matches notebook chart 4."""
    if tdf.empty or len(tdf) < window:
        return go.Figure()
    rolling_wr = (tdf['net_pnl'] > 0).rolling(window=window).mean() * 100
    avg_wr = (tdf['net_pnl'] > 0).mean() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(tdf) + 1)), y=rolling_wr,
        mode='lines', name=f'{window}-Trade Win Rate',
        line=dict(color=BLUE, width=2.5),
    ))
    fig.add_hline(y=avg_wr, line_dash='dash', line_color=RED, opacity=0.7,
                  annotation_text=f'Average {avg_wr:.1f}%')
    return _apply_layout(fig, f'Rolling Win Rate ({window}-Trade Window)', height=420,
                         xaxis_title='Trade Number', yaxis_title='Win Rate (%)')


# ============================================================
# 5. Trade Distribution Scatter
# ============================================================
def plot_trade_scatter(tdf):
    """Trade P&L scatter — matches notebook chart 5."""
    if tdf.empty:
        return go.Figure()
    wins = tdf[tdf['net_pnl'] > 0]
    losses = tdf[tdf['net_pnl'] <= 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wins.index, y=wins['net_pnl'], mode='markers',
        name='Wins', marker=dict(color=GREEN, size=7, opacity=0.6),
    ))
    fig.add_trace(go.Scatter(
        x=losses.index, y=losses['net_pnl'], mode='markers',
        name='Losses', marker=dict(color=RED, size=7, opacity=0.6),
    ))
    fig.add_hline(y=0, line_color='white', opacity=0.5)
    return _apply_layout(fig, 'Trade P&L Scatter Plot', height=450,
                         xaxis_title='Trade Number', yaxis_title='P&L (Rs)')


# ============================================================
# 6. Exit Reasons Pie
# ============================================================
def plot_exit_reasons_pie(tdf):
    """Exit reasons pie — matches notebook chart 6."""
    if tdf.empty:
        return go.Figure()
    exit_counts = tdf['exit_reason'].value_counts()
    colors = {'STOP_LOSS': RED, 'TRAILING_STOP': PURPLE, 'TIME_EXIT': ORANGE,
              'REGIME_CHANGE': BLUE, 'TARGET': GREEN}
    fig = go.Figure(go.Pie(
        labels=exit_counts.index, values=exit_counts.values,
        marker=dict(
            colors=[colors.get(r, '#888') for r in exit_counts.index],
            line=dict(color=DARK, width=3),
        ),
        textinfo='percent+label', textfont=dict(size=12),
        hole=0.4,
    ))
    return _apply_layout(fig, 'Trade Exit Reasons', height=420)


# ============================================================
# 7. Exit Reasons Bar
# ============================================================
def plot_exit_reasons_bar(tdf):
    """Exit reason breakdown bar chart."""
    if tdf.empty:
        return go.Figure()
    exit_colors = {'STOP_LOSS': RED, 'TRAILING_STOP': PURPLE, 'TIME_EXIT': ORANGE,
                   'REGIME_CHANGE': BLUE, 'TARGET': GREEN}
    counts = tdf['exit_reason'].value_counts()
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[exit_colors.get(r, '#888') for r in counts.index],
        text=counts.values, textposition='auto', textfont=dict(size=14),
    ))
    return _apply_layout(fig, 'Exit Reason Breakdown', height=400,
                         xaxis_title='Exit Reason', yaxis_title='Count')


# ============================================================
# 8. Regime Performance Bar
# ============================================================
def plot_regime_pnl(tdf, regime_map=None):
    """P&L by market regime — matches notebook chart 7."""
    if tdf.empty:
        return go.Figure()
    if regime_map is None:
        regime_map = REGIME_MAP
    regime_perf = tdf.groupby('regime')['net_pnl'].sum()
    labels = [regime_map.get(i, f'R{i}') for i in regime_perf.index]
    colors = [REGIME_COLORS.get(i, '#888') for i in regime_perf.index]

    fig = go.Figure(go.Bar(
        x=labels, y=regime_perf.values, marker_color=colors,
        text=[f'Rs {v:,.0f}' for v in regime_perf.values],
        textposition='auto',
    ))
    fig.add_hline(y=0, line_color='white', opacity=0.5)
    return _apply_layout(fig, 'Absolute P&L by Market Regime', height=450,
                         xaxis_title='Regime', yaxis_title='Total Net P&L (Rs)')


# ============================================================
# 9. Hold Time Distribution
# ============================================================
def plot_hold_distribution(tdf):
    """Hold time distribution — matches notebook chart 8."""
    if tdf.empty:
        return go.Figure()
    fig = go.Figure(go.Histogram(
        x=tdf['bars_held'], nbinsx=12,
        marker_color=PURPLE, opacity=0.8,
    ))
    fig.add_vline(x=tdf['bars_held'].mean(), line_dash='dash', line_color=GOLD,
                  annotation_text=f'Avg {tdf["bars_held"].mean():.1f} bars')
    return _apply_layout(fig, 'Trade Duration Distribution (5-Min Bars)', height=400,
                         xaxis_title='Bars Held', yaxis_title='Number of Trades')


# ============================================================
# 10. Hourly Performance
# ============================================================
def plot_hourly_pnl(tdf):
    """Net profit by hour — matches notebook chart 9."""
    if tdf.empty:
        return go.Figure()
    tmp = tdf.copy()
    tmp['hour'] = pd.to_datetime(tmp['entry_time']).dt.hour
    hour_perf = tmp.groupby('hour')['net_pnl'].sum()

    colors = [GREEN if v > 0 else RED for v in hour_perf.values]
    fig = go.Figure(go.Bar(
        x=hour_perf.index, y=hour_perf.values, marker_color=colors,
        text=[f'Rs {v:,.0f}' for v in hour_perf.values],
        textposition='auto',
    ))
    fig.add_hline(y=0, line_color='white', opacity=0.5)
    return _apply_layout(fig, 'Net Profit by Hour of Day', height=420,
                         xaxis_title='Hour (24H)', yaxis_title='Total P&L (Rs)')


# ============================================================
# 11. Position Size vs P&L
# ============================================================
def plot_position_vs_pnl(tdf):
    """Position size vs P&L — matches notebook chart 10."""
    if tdf.empty:
        return go.Figure()
    fig = go.Figure()
    for regime in sorted(tdf['regime'].unique()):
        sub = tdf[tdf['regime'] == regime]
        fig.add_trace(go.Scatter(
            x=sub['lots'], y=sub['net_pnl'], mode='markers',
            name=REGIME_MAP.get(regime, f'R{regime}'),
            marker=dict(color=REGIME_COLORS.get(regime, '#888'), size=8, opacity=0.7),
        ))
    fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.5)
    return _apply_layout(fig, 'Position Size vs Trade Outcome', height=450,
                         xaxis_title='Lots Deployed', yaxis_title='Net P&L (Rs)')


# ============================================================
# 12. Feature Importance
# ============================================================
def plot_feature_importance(feat_imp, top_n=10):
    """Top feature importances — matches notebook chart 11."""
    if feat_imp is None or (hasattr(feat_imp, 'empty') and feat_imp.empty):
        return go.Figure()
    if isinstance(feat_imp, dict):
        feat_imp = pd.Series(feat_imp)
    top = feat_imp.sort_values(ascending=False).head(top_n)
    fig = go.Figure(go.Bar(
        x=top.values[::-1], y=top.index[::-1], orientation='h',
        marker_color='#17becf', opacity=0.85,
        text=[f'{v:.4f}' for v in top.values[::-1]], textposition='auto',
    ))
    return _apply_layout(fig, f'Top {top_n} ML Feature Importances', height=max(350, top_n * 35),
                         xaxis_title='Relative Importance', yaxis_title='Feature')


# ============================================================
# 13. ML Signal Distribution
# ============================================================
def plot_signal_distribution(df_feat, min_signal_prob):
    """ML signal distribution — matches notebook chart 12."""
    if df_feat is None or 'ml_signal' not in df_feat.columns:
        return go.Figure()
    fig = go.Figure(go.Histogram(
        x=df_feat['ml_signal'], nbinsx=50,
        marker_color='indigo', opacity=0.8,
    ))
    fig.add_vline(x=min_signal_prob, line_dash='dash', line_color=RED, line_width=2,
                  annotation_text=f'Threshold ({min_signal_prob})')
    return _apply_layout(fig, 'ML Signal Probability Distribution', height=420,
                         xaxis_title='Signal Probability', yaxis_title='Frequency (Bars)')


# ============================================================
# 14. Regime Transition Matrix
# ============================================================
def plot_regime_transitions(df_feat):
    """Regime transition heatmap — matches notebook chart 13."""
    if df_feat is None or 'market_regime' not in df_feat.columns:
        return go.Figure()
    transition = pd.crosstab(
        df_feat['market_regime'].shift(), df_feat['market_regime'], normalize='index'
    )
    labels = [REGIME_MAP.get(i, f'R{i}') for i in range(4)]
    fig = go.Figure(go.Heatmap(
        z=transition.values * 100,
        x=labels, y=labels,
        colorscale='Blues',
        text=[[f'{v:.1f}%' for v in row] for row in transition.values * 100],
        texttemplate='%{text}', textfont=dict(size=12),
        colorbar=dict(title='Probability %'),
    ))
    return _apply_layout(fig, 'Market Regime Transition Probabilities', height=450,
                         xaxis_title='To Regime', yaxis_title='From Regime')


# ============================================================
# 15. Monte Carlo Fan Chart
# ============================================================
def plot_monte_carlo(mc_results):
    """Monte Carlo fan chart."""
    paths = mc_results.get('equity_paths', np.array([]))
    if paths.size == 0:
        return go.Figure()
    n_trades = paths.shape[1] - 1
    x = list(range(n_trades + 1))

    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=p95, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=p5, mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(64,196,255,0.1)',
                             name='5th-95th Percentile'))
    fig.add_trace(go.Scatter(x=x, y=p75, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=p25, mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(64,196,255,0.2)',
                             name='25th-75th Percentile'))
    fig.add_trace(go.Scatter(x=x, y=p50, mode='lines',
                             name='Median', line=dict(color=GOLD, width=2.5)))

    # Overlay actual strategy path
    actual = mc_results.get('actual_path', None)
    if actual is not None:
        fig.add_trace(go.Scatter(x=x, y=actual, mode='lines',
                                 name='Actual Strategy', line=dict(color=GREEN, width=3)))

    med = mc_results.get('median_final', 0)
    n_sims = mc_results.get('n_simulations', 0)
    return _apply_layout(fig,
                         f'Monte Carlo Simulation ({n_sims} paths)  |  Median: Rs {med:,.0f}',
                         height=500, xaxis_title='Trade #', yaxis_title='Cumulative P&L (Rs)')


# ============================================================
# 16. Per-Trade P&L Bar Chart
# ============================================================
def plot_per_trade_pnl(tdf):
    """Per-trade P&L bar chart."""
    if tdf.empty:
        return go.Figure()
    colors = [GREEN if v > 0 else RED for v in tdf['net_pnl']]
    fig = go.Figure(go.Bar(
        x=list(range(1, len(tdf) + 1)), y=tdf['net_pnl'],
        marker_color=colors, opacity=0.85,
    ))
    fig.add_hline(y=0, line_color='white', opacity=0.3)
    wr = (tdf['net_pnl'] > 0).mean() * 100
    return _apply_layout(fig, f'Net P&L per Trade  |  Win Rate {wr:.1f}%', height=450,
                         xaxis_title='Trade #', yaxis_title='Net P&L (Rs)')


# ============================================================
# 17. Win/Loss Pie
# ============================================================
def plot_win_loss_pie(tdf):
    if tdf.empty:
        return go.Figure()
    wc = (tdf['net_pnl'] > 0).sum()
    lc = (tdf['net_pnl'] <= 0).sum()
    fig = go.Figure(go.Pie(
        labels=['Wins', 'Losses'], values=[wc, lc],
        marker=dict(colors=[GREEN, RED], line=dict(color=DARK, width=3)),
        textinfo='percent+value', textfont=dict(size=14), hole=0.4,
    ))
    return _apply_layout(fig, f'Win/Loss — {wc}W / {lc}L', height=380)


# ============================================================
# 18. PnL Distribution
# ============================================================
def plot_pnl_distribution(tdf):
    if tdf.empty:
        return go.Figure()
    fig = go.Figure(go.Histogram(
        x=tdf['net_pnl'], nbinsx=40,
        marker_color=BLUE, opacity=0.8,
    ))
    fig.add_vline(x=0, line_color='white', opacity=0.5)
    fig.add_vline(x=tdf['net_pnl'].mean(), line_dash='dash', line_color=GOLD,
                  annotation_text=f'Mean Rs {tdf["net_pnl"].mean():,.0f}')
    return _apply_layout(fig, 'Trade P&L Distribution', height=400,
                         xaxis_title='Net P&L (Rs)', yaxis_title='Frequency')


# ============================================================
# 19. P&L by Weekday
# ============================================================
def plot_pnl_by_weekday(tdf):
    if tdf.empty:
        return go.Figure()
    tmp = tdf.copy()
    tmp['weekday'] = pd.to_datetime(tmp['entry_date']).dt.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    wd = tmp.groupby('weekday')['net_pnl'].sum().reindex(order).fillna(0)
    colors = [GREEN if v > 0 else RED for v in wd.values]
    fig = go.Figure(go.Bar(
        x=wd.index, y=wd.values, marker_color=colors, opacity=0.85,
        text=[f'Rs {v:,.0f}' for v in wd.values], textposition='auto',
    ))
    return _apply_layout(fig, 'P&L by Day of Week', height=400,
                         xaxis_title='Weekday', yaxis_title='Total P&L (Rs)')


# ============================================================
# 20. Daily Trade Count
# ============================================================
def plot_daily_trade_count(tdf):
    if tdf.empty:
        return go.Figure()
    daily = tdf.groupby('entry_date').size().reset_index(name='count')
    daily['entry_date'] = pd.to_datetime(daily['entry_date'])
    fig = go.Figure(go.Bar(
        x=daily['entry_date'], y=daily['count'],
        marker_color=BLUE, opacity=0.7,
    ))
    fig.add_hline(y=daily['count'].mean(), line_dash='dash', line_color=GOLD,
                  annotation_text=f'Avg {daily["count"].mean():.1f}')
    return _apply_layout(fig, 'Daily Trade Count', height=380,
                         xaxis_title='Date', yaxis_title='Trades')


# ============================================================
# 21. Rolling Sharpe
# ============================================================
def plot_rolling_sharpe(rolling_df, window=30):
    if rolling_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_df['date'], y=rolling_df['rolling_sharpe'],
        mode='lines', name=f'{window}-day Rolling Sharpe',
        line=dict(color=BLUE, width=2),
    ))
    fig.add_hline(y=0, line_color='white', opacity=0.3)
    fig.add_hline(y=1, line_dash='dash', line_color=GREEN, opacity=0.5, annotation_text='Sharpe = 1')
    fig.add_hline(y=2, line_dash='dash', line_color=GOLD, opacity=0.5, annotation_text='Sharpe = 2')
    return _apply_layout(fig, f'{window}-Day Rolling Sharpe Ratio', height=400,
                         xaxis_title='Date', yaxis_title='Sharpe Ratio')


# ============================================================
# 22. Stress Test Chart
# ============================================================
def plot_stress_test(df, x_col, y_col, title):
    if df.empty:
        return go.Figure()
    colors = [GREEN if v > 0 else RED for v in df[y_col]]
    fig = go.Figure(go.Bar(
        x=df[x_col], y=df[y_col], marker_color=colors, opacity=0.85,
        text=[f'Rs {v:,.0f}' for v in df[y_col]], textposition='auto',
    ))
    return _apply_layout(fig, title, height=400,
                         xaxis_title=x_col.replace('_', ' ').title(),
                         yaxis_title='Total Net P&L (Rs)')


# ============================================================
# 23. Regime Performance (generic)
# ============================================================
def plot_regime_performance(perf_df, title='Regime Performance'):
    if perf_df.empty:
        return go.Figure()
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Total P&L', 'Win Rate (%)'])
    colors = [GREEN if v > 0 else RED for v in perf_df['total_pnl']]
    fig.add_trace(go.Bar(
        x=perf_df.index, y=perf_df['total_pnl'], marker_color=colors,
        text=[f'Rs {v:,.0f}' for v in perf_df['total_pnl']], textposition='auto',
        name='P&L',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=perf_df.index, y=perf_df['win_rate'], marker_color=BLUE,
        text=[f'{v:.1f}%' for v in perf_df['win_rate']], textposition='auto',
        name='Win Rate',
    ), row=1, col=2)
    return _apply_layout(fig, title, height=420)


# ============================================================
# 24. Sensitivity Heatmap
# ============================================================
def plot_sensitivity_heatmap(df):
    if df.empty:
        return go.Figure()
    fig = go.Figure(go.Heatmap(
        z=df.values, x=df.columns.tolist(), y=df.index.tolist(),
        colorscale=[[0, RED], [0.5, DARK], [1, GREEN]],
        text=[[f'Rs {v:,.0f}' for v in row] for row in df.values],
        texttemplate='%{text}', textfont=dict(size=10),
        colorbar=dict(title='Net P&L'),
    ))
    return _apply_layout(fig, 'Parameter Sensitivity (Stop vs Signal)', height=450)


# ============================================================
# 25. Day Replay
# ============================================================
def plot_day_replay(day_data):
    """Intraday price chart with trade markers."""
    bars = day_data.get('day_bars', pd.DataFrame())
    trades = day_data.get('day_trades', pd.DataFrame())
    if bars.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bars['datetime'], y=bars['close'],
        mode='lines', name='Spot Price', line=dict(color=BLUE, width=2),
    ))

    if not trades.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(trades['entry_time']),
            y=trades['entry_price'],
            mode='markers', name='Entry',
            marker=dict(color=GREEN, size=14, symbol='triangle-up',
                        line=dict(color='white', width=1)),
        ))
        exit_colors = []
        for r in trades['exit_reason']:
            if r in ('STOP_LOSS', 'TRAILING_STOP'):
                exit_colors.append(RED)
            else:
                exit_colors.append(GOLD)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(trades['exit_time']),
            y=trades['exit_price'],
            mode='markers', name='Exit',
            marker=dict(color=exit_colors, size=14, symbol='triangle-down',
                        line=dict(color='white', width=1)),
            text=trades['exit_reason'],
            hovertemplate='Exit (%{text}) @ Rs %{y:.2f}<extra></extra>',
        ))

    date_str = day_data.get('summary', {}).get('date', '')
    pnl = day_data.get('summary', {}).get('day_pnl', 0)
    return _apply_layout(fig, f'Day Replay — {date_str}  |  P&L: Rs {pnl:,.2f}', height=500,
                         xaxis_title='Time', yaxis_title='Spot Price')


# ============================================================
# 26. Walk-Forward Fold Accuracy
# ============================================================
def plot_fold_accuracy(fold_results):
    """Walk-forward fold accuracy bar chart."""
    if not fold_results:
        return go.Figure()
    folds = [f['fold'] for f in fold_results]
    accs = [f['accuracy'] for f in fold_results]
    avg_acc = np.mean(accs)

    fig = go.Figure(go.Bar(
        x=[f'Fold {f}' for f in folds], y=accs,
        marker_color=BLUE, opacity=0.85,
        text=[f'{a:.4f}' for a in accs], textposition='auto',
    ))
    fig.add_hline(y=avg_acc, line_dash='dash', line_color=GOLD,
                  annotation_text=f'Avg {avg_acc:.4f}')
    return _apply_layout(fig, 'Walk-Forward Out-of-Sample Accuracy', height=400,
                         xaxis_title='Fold', yaxis_title='Accuracy')
