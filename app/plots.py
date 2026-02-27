"""
plots.py — All Plotly interactive charts.
Recreates every notebook matplotlib chart in Plotly with the same dark theme.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Color palette (matching notebook)
GOLD   = '#FFD700'
GREEN  = '#00E676'
RED    = '#FF5252'
DARK   = '#0d1117'
BLUE   = '#40C4FF'
ORANGE = '#FF9800'
PURPLE = '#CE93D8'
LOT_COLORS = {1: ORANGE, 2: GREEN, 3: BLUE}

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


def _apply_layout(fig: go.Figure, title: str = '', height: int = 500, **kwargs) -> go.Figure:
    layout = {**_LAYOUT, 'title': dict(text=title, font=dict(color=GOLD, size=16), x=0.5), 'height': height}
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


# ============================================================
# 1. Cumulative P&L + Drawdown
# ============================================================

def plot_cumulative_pnl(tdf: pd.DataFrame, metrics: dict) -> go.Figure:
    """Cumulative P&L with drawdown subplot. Based on Cell 10."""
    if tdf.empty:
        return go.Figure()
    xd = pd.to_datetime(tdf['exit_datetime'])
    cum = metrics.get('cum_pnl', tdf['net_pnl'].cumsum())
    dd  = metrics.get('dd_series', cum - cum.cummax())

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.05)

    # Cumulative P&L
    fig.add_trace(go.Scatter(
        x=xd, y=cum, mode='lines', name='Cumulative P&L',
        line=dict(color=GREEN, width=2),
        fill='tozeroy', fillcolor='rgba(0,230,118,0.08)',
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3, row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=xd, y=dd, mode='lines', name='Drawdown',
        line=dict(color=RED, width=1.5),
        fill='tozeroy', fillcolor='rgba(255,82,82,0.15)',
    ), row=2, col=1)

    net = metrics.get('net_pnl', cum.iloc[-1] if len(cum) > 0 else 0)
    sharpe = metrics.get('sharpe', 0)
    roi = metrics.get('roi', 0)
    title = f'Cumulative Net P&L  |  {len(tdf)} trades  |  Sharpe {sharpe}  |  ROI {roi}%'
    fig = _apply_layout(fig, title, height=550)
    fig.update_yaxes(title_text='Net P&L (₹)', row=1, col=1)
    fig.update_yaxes(title_text='Drawdown (₹)', row=2, col=1)
    return fig


# ============================================================
# 2. Per-Trade P&L Bar Chart
# ============================================================

def plot_per_trade_pnl(tdf: pd.DataFrame) -> go.Figure:
    """Per-trade P&L bar chart colored by lot size. Based on Cell 11."""
    if tdf.empty:
        return go.Figure()
    colors = [LOT_COLORS.get(lot, PURPLE) for lot in tdf['lots']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(tdf)+1)), y=tdf['net_pnl'],
        marker_color=colors, opacity=0.85,
        hovertemplate='Trade #%{x}<br>P&L: ₹%{y:,.2f}<extra></extra>',
    ))
    avg_w = tdf[tdf['net_pnl']>0]['net_pnl'].mean() if (tdf['net_pnl']>0).any() else 0
    avg_l = tdf[tdf['net_pnl']<=0]['net_pnl'].mean() if (tdf['net_pnl']<=0).any() else 0
    fig.add_hline(y=0, line_color='white', opacity=0.3)
    fig.add_hline(y=avg_w, line_dash='dash', line_color=GREEN, opacity=0.5,
                  annotation_text=f'Avg Win ₹{avg_w:,.0f}')
    fig.add_hline(y=avg_l, line_dash='dash', line_color=RED, opacity=0.5,
                  annotation_text=f'Avg Loss ₹{avg_l:,.0f}')
    wr = (tdf['net_pnl'] > 0).mean() * 100
    return _apply_layout(fig, f'Net P&L per Trade  |  Win Rate {wr:.1f}%', height=450,
                         xaxis_title='Trade #', yaxis_title='Net P&L (₹)')


# ============================================================
# 3. Win/Loss Pie Chart
# ============================================================

def plot_win_loss_pie(tdf: pd.DataFrame) -> go.Figure:
    """Win/Loss pie chart. Based on Cell 11."""
    if tdf.empty:
        return go.Figure()
    wc = (tdf['net_pnl'] > 0).sum()
    lc = (tdf['net_pnl'] <= 0).sum()
    fig = go.Figure(go.Pie(
        labels=['Wins', 'Losses'], values=[wc, lc],
        marker=dict(colors=[GREEN, RED], line=dict(color=DARK, width=3)),
        textinfo='percent+value', textfont=dict(size=14),
        hole=0.4,
    ))
    return _apply_layout(fig, f'Win/Loss  —  {wc}W / {lc}L', height=380)


# ============================================================
# 4. Monthly P&L Heatmap
# ============================================================

def plot_monthly_heatmap(piv: pd.DataFrame) -> go.Figure:
    """Monthly P&L heatmap. Based on Cell 12."""
    if piv.empty:
        return go.Figure()
    max_val = np.abs(piv.values).max()
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
        colorscale=[[0, RED], [0.5, DARK], [1, GREEN]],
        zmin=-max_val, zmax=max_val,
        text=[[f'₹{v:,.0f}' for v in row] for row in piv.values],
        texttemplate='%{text}', textfont=dict(size=12, color='white'),
        hovertemplate='Year: %{y}<br>Month: %{x}<br>P&L: %{text}<extra></extra>',
        colorbar=dict(title='P&L (₹)', tickformat=','),
    ))
    return _apply_layout(fig, 'Monthly Net P&L Heatmap', height=max(350, len(piv) * 80))


# ============================================================
# 5. Entry IV vs P&L Scatter
# ============================================================

def plot_iv_vs_pnl(tdf: pd.DataFrame) -> go.Figure:
    """Entry IV vs Net P&L scatter. Based on Cell 13."""
    if tdf.empty:
        return go.Figure()
    fig = go.Figure()
    for lots in sorted(tdf['lots'].unique()):
        sub = tdf[tdf['lots'] == lots]
        fig.add_trace(go.Scatter(
            x=sub['entry_iv'], y=sub['net_pnl'], mode='markers',
            name=f'{lots} lot', marker=dict(color=LOT_COLORS.get(lots, PURPLE), size=7, opacity=0.7),
            hovertemplate='IV: %{x:.2f}%<br>P&L: ₹%{y:,.2f}<extra></extra>',
        ))
    # Trend line
    if len(tdf) > 1:
        z = np.polyfit(tdf['entry_iv'], tdf['net_pnl'], 1)
        x_range = np.linspace(tdf['entry_iv'].min(), tdf['entry_iv'].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_range, y=np.polyval(z, x_range), mode='lines',
            name=f'Trend (slope={z[0]:.2f})', line=dict(color=GOLD, dash='dash', width=2),
        ))
    corr = tdf['entry_iv'].corr(tdf['net_pnl'])
    return _apply_layout(fig, f'Entry IV vs Net P&L  |  Corr: {corr:.3f}', height=480,
                         xaxis_title='Entry IV (%)', yaxis_title='Net P&L (₹)')


# ============================================================
# 6. Hold Duration by Lot Size
# ============================================================

def plot_hold_duration(tdf: pd.DataFrame, max_hold: int = 9) -> go.Figure:
    """Hold duration distribution by lot size. Based on Cell 14."""
    if tdf.empty:
        return go.Figure()
    fig = go.Figure()
    for lots in sorted(tdf['lots'].unique()):
        sub = tdf[tdf['lots'] == lots]
        counts = [sub[sub['bars_held'] == i].shape[0] for i in range(1, max_hold + 1)]
        fig.add_trace(go.Bar(
            x=list(range(1, max_hold + 1)), y=counts,
            name=f'{lots} lot', marker_color=LOT_COLORS.get(lots, PURPLE), opacity=0.85,
        ))
    fig.add_vline(x=tdf['bars_held'].mean(), line_dash='dash', line_color=GOLD,
                  annotation_text=f'Avg {tdf["bars_held"].mean():.1f} bars')
    return _apply_layout(fig, 'Hold Duration Distribution by Lot Size', height=420,
                         xaxis_title='Bars Held', yaxis_title='Number of Trades',
                         barmode='group')


# ============================================================
# 7. ML Signal Distribution
# ============================================================

def plot_ml_signal_dist(tdf: pd.DataFrame, ml_threshold: float = 0.46) -> go.Figure:
    """ML signal distribution by lot size. Based on Cells 15/18."""
    if tdf.empty:
        return go.Figure()
    fig = go.Figure()
    for lots in sorted(tdf['lots'].unique()):
        sub = tdf[tdf['lots'] == lots]
        fig.add_trace(go.Histogram(
            x=sub['ml_signal'], name=f'{lots} lot (n={len(sub)})',
            marker_color=LOT_COLORS.get(lots, PURPLE), opacity=0.7,
            nbinsx=25,
        ))
    fig.add_vline(x=ml_threshold, line_dash='dash', line_color=GOLD, line_width=2,
                  annotation_text=f'Threshold = {ml_threshold:.2f}')
    return _apply_layout(fig, 'ML Signal Distribution by Lot Size', height=420,
                         xaxis_title='ML Confidence Score', yaxis_title='Count',
                         barmode='overlay')


# ============================================================
# 8. Exit Reasons
# ============================================================

def plot_exit_reasons(tdf: pd.DataFrame) -> go.Figure:
    """Exit reason breakdown. Based on Cell 16."""
    if tdf.empty:
        return go.Figure()
    exit_colors = {'TARGET': GREEN, 'STOP': RED, 'TIME': ORANGE, 'DAY_END': BLUE, 'TRAIL_STOP': PURPLE}
    counts = tdf['exit_reason'].value_counts()
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[exit_colors.get(r, '#888') for r in counts.index],
        text=counts.values, textposition='auto', textfont=dict(size=14),
    ))
    return _apply_layout(fig, 'Exit Reason Breakdown', height=400,
                         xaxis_title='Exit Reason', yaxis_title='Count')


# ============================================================
# 9. Equity Curve (annotated)
# ============================================================

def plot_equity_curve(equity_curve: list, tdf: pd.DataFrame, cfg: dict, metrics: dict) -> go.Figure:
    """Equity curve with annotations. Based on Cell 20."""
    eq_df = pd.DataFrame(equity_curve, columns=['datetime', 'capital']).dropna()
    if eq_df.empty:
        return go.Figure()
    eq_df['datetime'] = pd.to_datetime(eq_df['datetime'])
    INITIAL = cfg['INITIAL_CAPITAL']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_df['datetime'], y=eq_df['capital'], mode='lines',
        name='Capital', line=dict(color=GREEN, width=2.5),
        fill='tozeroy', fillcolor='rgba(0,230,118,0.06)',
    ))
    fig.add_hline(y=INITIAL, line_dash='dash', line_color='white', opacity=0.5,
                  annotation_text=f'Start ₹{INITIAL:,}')

    # Peak / Trough / Final
    peak_idx = eq_df['capital'].idxmax()
    trough_idx = eq_df['capital'].idxmin()
    fig.add_trace(go.Scatter(
        x=[eq_df.loc[peak_idx, 'datetime']], y=[eq_df.loc[peak_idx, 'capital']],
        mode='markers', name=f'Peak ₹{eq_df.loc[peak_idx, "capital"]:,.0f}',
        marker=dict(color=GREEN, size=12, symbol='diamond'),
    ))
    fig.add_trace(go.Scatter(
        x=[eq_df.loc[trough_idx, 'datetime']], y=[eq_df.loc[trough_idx, 'capital']],
        mode='markers', name=f'Trough ₹{eq_df.loc[trough_idx, "capital"]:,.0f}',
        marker=dict(color=RED, size=12, symbol='diamond'),
    ))
    fig.add_trace(go.Scatter(
        x=[eq_df.iloc[-1]['datetime']], y=[eq_df.iloc[-1]['capital']],
        mode='markers', name=f'Final ₹{eq_df.iloc[-1]["capital"]:,.0f}',
        marker=dict(color=GOLD, size=14, symbol='star'),
    ))

    net = metrics.get('net_pnl', 0)
    roi = metrics.get('roi', 0)
    return _apply_layout(fig, f'Equity Curve  —  Final P&L: ₹{net:,.0f} (ROI: {roi}%)', height=500,
                         xaxis_title='Date', yaxis_title='Capital (₹)')


# ============================================================
# 10. PnL Distribution Histogram
# ============================================================

def plot_pnl_distribution(tdf: pd.DataFrame) -> go.Figure:
    """Trade P&L distribution histogram."""
    if tdf.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=tdf['net_pnl'], nbinsx=40,
        marker_color=BLUE, opacity=0.8,
        hovertemplate='P&L Range: ₹%{x}<br>Count: %{y}<extra></extra>',
    ))
    fig.add_vline(x=0, line_color='white', opacity=0.5)
    fig.add_vline(x=tdf['net_pnl'].mean(), line_dash='dash', line_color=GOLD,
                  annotation_text=f'Mean ₹{tdf["net_pnl"].mean():,.0f}')
    return _apply_layout(fig, 'Trade P&L Distribution', height=400,
                         xaxis_title='Net P&L (₹)', yaxis_title='Frequency')


# ============================================================
# 11. PnL by Weekday
# ============================================================

def plot_pnl_by_weekday(tdf: pd.DataFrame) -> go.Figure:
    """P&L breakdown by day of week."""
    if tdf.empty:
        return go.Figure()
    tmp = tdf.copy()
    tmp['weekday'] = pd.to_datetime(tmp['date']).dt.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    wd = tmp.groupby('weekday')['net_pnl'].sum().reindex(order).fillna(0)
    colors = [GREEN if v > 0 else RED for v in wd.values]
    fig = go.Figure(go.Bar(
        x=wd.index, y=wd.values, marker_color=colors, opacity=0.85,
        text=[f'₹{v:,.0f}' for v in wd.values], textposition='auto',
    ))
    return _apply_layout(fig, 'P&L by Day of Week', height=400,
                         xaxis_title='Weekday', yaxis_title='Total P&L (₹)')


# ============================================================
# 12. Day Replay Chart
# ============================================================

def plot_day_replay(day_data: dict) -> go.Figure:
    """Intraday straddle price chart with trade markers."""
    bars = day_data.get('day_bars', pd.DataFrame())
    trades = day_data.get('day_trades', pd.DataFrame())
    if bars.empty:
        return go.Figure()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.08,
                        subplot_titles=['Straddle Price', 'IV'])

    # Straddle price
    fig.add_trace(go.Scatter(
        x=bars['datetime'], y=bars['Straddle_Price'],
        mode='lines', name='Straddle Price', line=dict(color=BLUE, width=2),
    ), row=1, col=1)

    # Entry/Exit markers
    if not trades.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(trades['entry_datetime']),
            y=trades['entry_price'],
            mode='markers', name='Entry',
            marker=dict(color=GREEN, size=14, symbol='triangle-up', line=dict(color='white', width=1)),
            hovertemplate='Entry @ ₹%{y:.2f}<extra></extra>',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(trades['exit_datetime']),
            y=trades['exit_price'],
            mode='markers', name='Exit',
            marker=dict(
                color=[RED if r in ('STOP', 'TRAIL_STOP') else GOLD for r in trades['exit_reason']],
                size=14, symbol='triangle-down', line=dict(color='white', width=1),
            ),
            text=trades['exit_reason'],
            hovertemplate='Exit (%{text}) @ ₹%{y:.2f}<extra></extra>',
        ), row=1, col=1)

    # IV subplot
    fig.add_trace(go.Scatter(
        x=bars['datetime'], y=bars['iv'],
        mode='lines', name='IV', line=dict(color=PURPLE, width=1.5),
    ), row=2, col=1)

    date_str = day_data.get('summary', {}).get('date', '')
    pnl = day_data.get('summary', {}).get('day_pnl', 0)
    return _apply_layout(fig, f'Day Replay — {date_str}  |  P&L: ₹{pnl:,.2f}', height=600)


# ============================================================
# 13. IV Percentile Time Series
# ============================================================

def plot_iv_percentile(iv_data: pd.DataFrame) -> go.Figure:
    """IV percentile over time."""
    if iv_data.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iv_data['date'], y=iv_data['iv_pct'],
        mode='lines', name='IV Percentile', line=dict(color=PURPLE, width=1.5),
        fill='tozeroy', fillcolor='rgba(206,147,216,0.1)',
    ))
    fig.add_hline(y=80, line_dash='dash', line_color=RED, opacity=0.5,
                  annotation_text='High IV (80th)')
    fig.add_hline(y=20, line_dash='dash', line_color=GREEN, opacity=0.5,
                  annotation_text='Low IV (20th)')
    return _apply_layout(fig, 'IV Percentile Over Time', height=400,
                         xaxis_title='Date', yaxis_title='IV Percentile (%)')


# ============================================================
# 14. Monte Carlo Fan Chart
# ============================================================

def plot_monte_carlo(mc_results: dict) -> go.Figure:
    """Monte Carlo simulation fan chart."""
    paths = mc_results.get('equity_paths', np.array([]))
    if paths.size == 0:
        return go.Figure()
    n_trades = paths.shape[1] - 1
    x = list(range(n_trades + 1))

    # Compute percentiles
    p5  = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig = go.Figure()
    # 5-95 band
    fig.add_trace(go.Scatter(x=x, y=p95, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=p5, mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(64,196,255,0.1)',
                             name='5th-95th Percentile'))
    # 25-75 band
    fig.add_trace(go.Scatter(x=x, y=p75, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=p25, mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(64,196,255,0.2)',
                             name='25th-75th Percentile'))
    # Median
    fig.add_trace(go.Scatter(x=x, y=p50, mode='lines',
                             name='Median', line=dict(color=GOLD, width=2.5)))

    med = mc_results.get('median_final', 0)
    return _apply_layout(fig, f'Monte Carlo Simulation ({mc_results.get("n_simulations", 0)} paths)  |  Median: ₹{med:,.0f}',
                         height=500, xaxis_title='Trade #', yaxis_title='Capital (₹)')


# ============================================================
# 15. Rolling Sharpe
# ============================================================

def plot_rolling_sharpe(rolling_df: pd.DataFrame, window: int = 30) -> go.Figure:
    """Rolling Sharpe ratio chart."""
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
# 16. IV Change per Trade
# ============================================================

def plot_iv_change(tdf: pd.DataFrame) -> go.Figure:
    """IV change per trade by lot size. Based on Cell 19."""
    if tdf.empty:
        return go.Figure()
    fig = go.Figure()
    for lots in sorted(tdf['lots'].unique()):
        sub = tdf[tdf['lots'] == lots]
        fig.add_trace(go.Scatter(
            x=sub['trade_num'], y=sub['iv_change'], mode='markers',
            name=f'{lots} lot',
            marker=dict(color=LOT_COLORS.get(lots, PURPLE), size=6, opacity=0.6),
        ))
    fig.add_hline(y=0, line_color='white', opacity=0.3)
    fig.add_hline(y=tdf['iv_change'].mean(), line_dash='dash', line_color=GOLD,
                  annotation_text=f'Avg {tdf["iv_change"].mean():.2f}%')
    return _apply_layout(fig, 'IV Change per Trade (Exit - Entry)', height=420,
                         xaxis_title='Trade #', yaxis_title='IV Change (%)')


# ============================================================
# 17. Feature Importance
# ============================================================

def plot_feature_importance(feat_imp: pd.Series, top_n: int = 15) -> go.Figure:
    """Top feature importance bar chart. Based on Cell 5."""
    if feat_imp is None or feat_imp.empty:
        return go.Figure()
    top = feat_imp.head(top_n)
    fig = go.Figure(go.Bar(
        x=top.values[::-1], y=top.index[::-1], orientation='h',
        marker_color=BLUE, opacity=0.85,
        text=[f'{v:.4f}' for v in top.values[::-1]], textposition='auto',
    ))
    return _apply_layout(fig, f'Top {top_n} Feature Importance (Random Forest)', height=max(350, top_n * 30),
                         xaxis_title='Importance', yaxis_title='Feature')


# ============================================================
# 18. Stress Test Charts
# ============================================================

def plot_stress_test(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    """Generic stress test bar chart."""
    if df.empty:
        return go.Figure()
    colors = [GREEN if v > 0 else RED for v in df[y_col]]
    fig = go.Figure(go.Bar(
        x=df[x_col], y=df[y_col], marker_color=colors, opacity=0.85,
        text=[f'₹{v:,.0f}' for v in df[y_col]], textposition='auto',
    ))
    return _apply_layout(fig, title, height=400,
                         xaxis_title=x_col.replace('_', ' ').title(),
                         yaxis_title='Total Net P&L (₹)')


# ============================================================
# 19. Regime Performance Chart
# ============================================================

def plot_regime_performance(perf_df: pd.DataFrame, title: str = 'Regime Performance') -> go.Figure:
    """Regime performance chart with grouped metrics."""
    if perf_df.empty:
        return go.Figure()
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Total P&L', 'Win Rate (%)'])
    colors = [GREEN if v > 0 else RED for v in perf_df['total_pnl']]
    fig.add_trace(go.Bar(
        x=perf_df.index, y=perf_df['total_pnl'], marker_color=colors,
        text=[f'₹{v:,.0f}' for v in perf_df['total_pnl']], textposition='auto',
        name='P&L',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=perf_df.index, y=perf_df['win_rate'], marker_color=BLUE,
        text=[f'{v:.1f}%' for v in perf_df['win_rate']], textposition='auto',
        name='Win Rate',
    ), row=1, col=2)
    return _apply_layout(fig, title, height=420)


# ============================================================
# 20. Holding Time Distribution
# ============================================================

def plot_holding_distribution(tdf: pd.DataFrame) -> go.Figure:
    """Holding time distribution histogram."""
    if tdf.empty:
        return go.Figure()
    fig = go.Figure(go.Histogram(
        x=tdf['bars_held'] * 5, nbinsx=20,
        marker_color=ORANGE, opacity=0.8,
    ))
    fig.add_vline(x=tdf['bars_held'].mean() * 5, line_dash='dash', line_color=GOLD,
                  annotation_text=f'Avg {tdf["bars_held"].mean()*5:.0f} min')
    return _apply_layout(fig, 'Holding Time Distribution', height=380,
                         xaxis_title='Holding Time (minutes)', yaxis_title='Count')


# ============================================================
# 21. Daily Trade Count
# ============================================================

def plot_daily_trade_count(tdf: pd.DataFrame) -> go.Figure:
    """Daily number of trades."""
    if tdf.empty:
        return go.Figure()
    daily = tdf.groupby('date').size().reset_index(name='count')
    daily['date'] = pd.to_datetime(daily['date'])
    fig = go.Figure(go.Bar(
        x=daily['date'], y=daily['count'],
        marker_color=BLUE, opacity=0.7,
    ))
    fig.add_hline(y=daily['count'].mean(), line_dash='dash', line_color=GOLD,
                  annotation_text=f'Avg {daily["count"].mean():.1f}')
    return _apply_layout(fig, 'Daily Trade Count', height=380,
                         xaxis_title='Date', yaxis_title='Trades')


# ============================================================
# 22. Parameter Sensitivity Heatmap
# ============================================================

def plot_sensitivity_heatmap(df: pd.DataFrame) -> go.Figure:
    """Stop vs Target sensitivity heatmap."""
    if df.empty:
        return go.Figure()
    fig = go.Figure(go.Heatmap(
        z=df.values, x=df.columns.tolist(), y=df.index.tolist(),
        colorscale=[[0, RED], [0.5, DARK], [1, GREEN]],
        text=[[f'₹{v:,.0f}' for v in row] for row in df.values],
        texttemplate='%{text}', textfont=dict(size=10),
        colorbar=dict(title='Net P&L'),
    ))
    return _apply_layout(fig, 'Parameter Sensitivity  —  Target (%) vs Stop (%)', height=450,
                         xaxis_title='Stop Loss (%)', yaxis_title='Target (%)')
