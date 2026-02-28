"""
main.py — Regime-Switch ML Bot Streamlit Dashboard
Exact reproduction of Final_straddle.ipynb strategy with all sections.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# ── Ensure app package is importable ──
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_config_dict, REGIME_MAP, REGIME_COLORS
from app.utils import inject_custom_css
from app.data_loader import load_and_clean, get_data_summary
from app.strategy import engineer_features, train_ml_models, run_backtest, get_feature_columns
from app.metrics import (
    compute_metrics, compute_monthly_breakdown, compute_monthly_heatmap_data,
    compute_exit_breakdown, compute_lot_performance, compute_regime_performance,
    compute_entry_quality, compute_weekday_pnl, compute_hourly_pnl,
)
from app.plots import (
    plot_equity_curve, plot_drawdown, plot_monthly_heatmap,
    plot_rolling_win_rate, plot_trade_scatter, plot_exit_reasons_pie,
    plot_exit_reasons_bar, plot_regime_pnl, plot_hold_distribution,
    plot_hourly_pnl, plot_position_vs_pnl, plot_feature_importance,
    plot_signal_distribution, plot_regime_transitions,
    plot_monte_carlo, plot_per_trade_pnl, plot_win_loss_pie,
    plot_pnl_distribution, plot_pnl_by_weekday, plot_daily_trade_count,
    plot_rolling_sharpe, plot_stress_test, plot_regime_performance,
    plot_sensitivity_heatmap, plot_day_replay, plot_fold_accuracy,
)
from app.monte_carlo import run_monte_carlo
from app.regime_analysis import (
    get_regime_distribution, get_regime_trade_performance,
    classify_vol_trend_regimes, get_regime_performance,
)
from app.trade_replay import get_day_data, get_available_dates
from app.iv_analytics import compute_vol_percentile_series, compute_vol_change_analysis
from app.robustness import (
    slippage_stress_test, brokerage_stress_test,
    compute_rolling_sharpe, parameter_sensitivity_heatmap,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title='Regime-Switch ML Bot',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded',
)
st.markdown(inject_custom_css(), unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR — CONTROL PANEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown('<h2 style="color:#F1F5F9;font-weight:600;">Control Panel</h2>', unsafe_allow_html=True)
    st.caption('Adjust parameters — backtest re-runs live')
    st.divider()

    # ── Capital & Position Sizing ──
    st.markdown('#### Capital & Sizing')
    initial_capital = st.number_input('Initial Capital (Rs)', 100_000, 50_000_000, 1_000_000, step=100_000)
    lot_size = st.number_input('Lot Size', 1, 500, 65, step=1)
    max_risk = st.slider('Max Risk Per Trade %', 0.5, 5.0, 1.0, 0.5, format='%.1f')

    st.divider()

    # ── Strategy Parameters ──
    st.markdown('#### Strategy Parameters')
    stop_loss_pct = st.slider('Stop Loss % (spot)', -0.005, -0.0005, -0.0015, 0.0005, format='%.4f')
    trailing_activation = st.slider('Trailing Activation %', 0.0005, 0.005, 0.0012, 0.0001, format='%.4f')
    trailing_distance = st.slider('Trailing Distance %', 0.0001, 0.003, 0.0005, 0.0001, format='%.4f')
    max_hold_bars = st.slider('Max Hold (bars)', 3, 30, 12)

    st.divider()

    # ── ML Parameters ──
    st.markdown('#### ML Parameters')
    min_signal_prob = st.slider('Min Signal Prob', 0.20, 0.60, 0.32, 0.01, format='%.2f')
    min_expected_return = st.slider('Min Expected Return', 0.00005, 0.001, 0.0001, 0.00005, format='%.5f')
    n_estimators = st.slider('RF Trees', 50, 500, 100, 50)
    max_depth = st.slider('RF Max Depth', 3, 15, 5)
    n_splits = st.slider('Walk-Forward Splits', 3, 10, 5)

    st.divider()

    # ── Cost Parameters ──
    st.markdown('#### Costs')
    slippage_rate = st.slider('Slippage Rate', 0.0, 0.005, 0.001, 0.0005, format='%.4f')
    brokerage = st.number_input('Brokerage (Rs/trade)', 0, 500, 40, step=10)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILD CONFIG FROM SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cfg = get_config_dict(
    INITIAL_CAPITAL=initial_capital,
    LOT_SIZE=lot_size,
    MAX_RISK_PER_TRADE=max_risk / 100.0,
    STOP_LOSS_PCT=stop_loss_pct,
    TRAILING_STOP_ACTIVATION=trailing_activation,
    TRAILING_STOP_DISTANCE=trailing_distance,
    MAX_HOLD_BARS=max_hold_bars,
    MIN_SIGNAL_PROB=min_signal_prob,
    MIN_EXPECTED_RETURN=min_expected_return,
    N_ESTIMATORS=n_estimators,
    MAX_DEPTH=max_depth,
    N_SPLITS=n_splits,
    SLIPPAGE_RATE=slippage_rate,
    BROKERAGE_PER_TRADE=brokerage,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING & PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(show_spinner=False)
def load_data():
    csv_path = ROOT / 'FINAL_NIFTY_MASTER_ATM_5min.csv'
    if not csv_path.exists():
        for alt in [ROOT / 'data' / 'FINAL_NIFTY_MASTER_ATM_5min.csv']:
            if alt.exists():
                csv_path = alt
                break
    return load_and_clean(csv_path)


@st.cache_data(show_spinner=False)
def run_feature_engineering(_df_raw, hv_short, hv_long, momentum_threshold):
    feat_cfg = {
        'HV_SHORT_WINDOW': hv_short,
        'HV_LONG_WINDOW': hv_long,
        'MOMENTUM_THRESHOLD': momentum_threshold,
    }
    return engineer_features(_df_raw, feat_cfg)


@st.cache_data(show_spinner=False)
def run_ml_training(_df_feat, n_splits, n_estimators, max_depth, random_state, min_signal_prob):
    ml_cfg = {
        'N_SPLITS': n_splits,
        'N_ESTIMATORS': n_estimators,
        'MAX_DEPTH': max_depth,
        'RANDOM_STATE': random_state,
        'MIN_SIGNAL_PROB': min_signal_prob,
    }
    return train_ml_models(_df_feat, ml_cfg)


# ── Execute Pipeline ──
pipeline_ok = False
tdf = pd.DataFrame()
summary = {}
metrics = {}
data_summary = {}
df_feat = pd.DataFrame()
models = []
scalers = []
feature_cols = []
fold_results = []

progress_bar = st.progress(0, text='Loading data...')
try:
    df_raw = load_data()
    data_summary = get_data_summary(df_raw)
    progress_bar.progress(15, text=f'Data loaded — {data_summary["total_rows"]:,} rows  |  Engineering features...')

    df_feat = run_feature_engineering(
        df_raw,
        cfg['HV_SHORT_WINDOW'],
        cfg['HV_LONG_WINDOW'],
        cfg['MOMENTUM_THRESHOLD'],
    )
    progress_bar.progress(35, text=f'Features built ({len(df_feat):,} rows)  |  Training ML...')

    df_feat, models, scalers, feature_cols, fold_results = run_ml_training(
        df_feat,
        cfg['N_SPLITS'],
        cfg['N_ESTIMATORS'],
        cfg['MAX_DEPTH'],
        cfg['RANDOM_STATE'],
        cfg['MIN_SIGNAL_PROB'],
    )

    avg_acc = np.mean([f['accuracy'] for f in fold_results])
    progress_bar.progress(65, text=f'ML trained (Avg Acc: {avg_acc:.4f})  |  Running backtest...')

    tdf, summary, risk_mgr = run_backtest(df_feat, cfg)
    if not tdf.empty:
        metrics = compute_metrics(tdf, cfg)
    pipeline_ok = True

    n_trades = summary.get('total_trades', 0)
    net_pnl = summary.get('total_pnl', 0)
    progress_bar.progress(100, text=f'Complete — {n_trades} trades  |  Net P&L: Rs {net_pnl:,.0f}')
except Exception as e:
    st.error(f'Pipeline error: {e}')
    import traceback
    st.code(traceback.format_exc())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div class="gradient-header">Regime-Switch ML Bot — NIFTY 5-Min Scalper</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Walk-Forward RandomForest  ·  4-Regime Classification  ·  Dynamic Position Sizing</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tabs = st.tabs([
    'Overview',
    'Results & Equity',
    'Trades',
    'Day Replay',
    'Regimes',
    'ML Analytics',
    'Monte Carlo',
    'Rolling Sharpe',
    'Robustness',
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: STRATEGY OVERVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[0]:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('#### Strategy Description')
        st.markdown("""
        **Regime-Switch ML Bot** — A NIFTY 5-minute scalping strategy using a RandomForest model
        trained via walk-forward (TimeSeriesSplit) to predict short-term price momentum.

        The strategy classifies market conditions into **4 regimes** based on volatility and trend:
        - **Trend HV** (High Vol trending) — standard sizing
        - **Trend LV** (Low Vol trending) — 1.2x sizing
        - **Range HV** (High Vol ranging) — 0.7x sizing
        - **Range LV** (Low Vol ranging) — **blocked** (no trades)

        **Key Innovation:** Dynamic position sizing based on ML confidence, regime classification,
        and anti-martingale consecutive loss penalty.
        """)

        st.markdown('#### Entry Conditions')
        st.markdown(f"""
        1. ML signal probability ≥ **{cfg['MIN_SIGNAL_PROB']:.2f}**
        2. Market regime ≠ Range LV (regime 3 blocked)
        3. Expected return (signal × 0.5%) > **{cfg['MIN_EXPECTED_RETURN']:.5f}**
        4. Dynamic lot sizing via RiskManager
        """)

        st.markdown('#### Exit Rules')
        st.markdown(f"""
        1. **Stop Loss:** Spot move < **{cfg['STOP_LOSS_PCT']*100:.2f}%**
        2. **Trailing Stop:** Activate at **+{cfg['TRAILING_STOP_ACTIVATION']*100:.2f}%**, trail by **{cfg['TRAILING_STOP_DISTANCE']*100:.2f}%**
        3. **Time Exit:** After **{cfg['MAX_HOLD_BARS']}** bars ({cfg['MAX_HOLD_BARS']*5} min)
        4. **Regime Change:** Exit with profit if regime changes
        """)

    with c2:
        st.markdown('#### Parameters')
        params_data = {
            'Parameter': ['Instrument', 'Timeframe', 'Type', 'Lot Size', 'ML Model',
                          'Walk-Forward Folds', 'Capital', 'Max Risk/Trade'],
            'Value': ['NIFTY 50 Spot', '5-min bars', 'Intraday Scalping',
                      str(cfg['LOT_SIZE']), f'RF({cfg["N_ESTIMATORS"]})',
                      str(cfg['N_SPLITS']),
                      f'Rs {cfg["INITIAL_CAPITAL"]:,}', f'{cfg["MAX_RISK_PER_TRADE"]*100:.1f}%'],
        }
        st.dataframe(pd.DataFrame(params_data), hide_index=True, use_container_width=True)

        # Data Summary
        st.markdown('#### Data Summary')
        if data_summary:
            st.metric('Period', f'{data_summary["start_date"]} → {data_summary["end_date"]}')
            c2a, c2b = st.columns(2)
            c2a.metric('Total Rows', f'{data_summary["total_rows"]:,}')
            c2b.metric('Trading Days', f'{data_summary["trading_days"]:,}')
            st.metric('Price Column', data_summary.get('price_col', 'spot'))

        # ML Stats
        if fold_results:
            st.markdown('#### Walk-Forward Performance')
            for fr in fold_results:
                st.metric(f'Fold {fr["fold"]} Accuracy', f'{fr["accuracy"]:.4f}')
            avg_acc = np.mean([f['accuracy'] for f in fold_results])
            st.metric('Average Accuracy', f'{avg_acc:.4f}')

        # Feature engineering summary
        if not df_feat.empty:
            st.markdown('#### Feature Summary')
            st.metric('Total Features', len(feature_cols))
            st.metric('Target Positive %', f'{df_feat["target"].mean()*100:.2f}%')
            regime_counts = df_feat['market_regime'].value_counts().sort_index()
            for i, count in regime_counts.items():
                st.metric(f'{REGIME_MAP.get(i, f"R{i}")}', f'{count:,} ({count/len(df_feat)*100:.1f}%)')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: BACKTEST RESULTS & EQUITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[1]:
    if not tdf.empty:
        st.markdown('### Key Metrics')

        # Row 1: Top-line metrics (matching notebook P&L ANALYSIS output)
        r1 = st.columns(5)
        r1[0].metric('Net P&L', f'Rs {metrics["net_pnl"]:,.0f}',
                      delta=f'{metrics["roi"]:.2f}% ROI')
        r1[1].metric('Win Rate', f'{metrics["win_rate"]:.2f}%',
                      delta=f'{metrics["winning_trades"]}W / {metrics["losing_trades"]}L')
        r1[2].metric('Profit Factor', f'{metrics["profit_factor"]:.2f}')
        r1[3].metric('Max Drawdown', f'{metrics["max_drawdown_pct"]:.2f}%')
        r1[4].metric('Final Capital', f'Rs {metrics["final_capital"]:,.0f}')

        st.divider()

        # Row 2: Detailed metrics
        r2 = st.columns(5)
        r2[0].metric('Total Trades', f'{metrics["total_trades"]}')
        r2[1].metric('Sharpe Ratio', f'{metrics["sharpe"]:.2f}')
        r2[2].metric('Sortino Ratio', f'{metrics["sortino"]:.2f}')
        r2[3].metric('Avg Hold', f'{metrics["avg_hold_min"]:.0f} min')
        r2[4].metric('Risk/Reward', f'{metrics["risk_reward"]:.2f}x')

        st.divider()

        # ── Equity Curve ──
        st.markdown('### Equity Curve')
        st.plotly_chart(plot_equity_curve(tdf, cfg, metrics), use_container_width=True)

        st.markdown('### Drawdown')
        st.plotly_chart(plot_drawdown(tdf, cfg), use_container_width=True)

        st.markdown('### Monthly P&L Heatmap')
        piv = compute_monthly_heatmap_data(tdf)
        if not piv.empty:
            st.plotly_chart(plot_monthly_heatmap(piv), use_container_width=True)

        st.divider()

        # Row 3: Cost breakdown
        r3 = st.columns(5)
        r3[0].metric('Gross P&L', f'Rs {metrics["gross_pnl"]:,.0f}')
        r3[1].metric('Slippage', f'Rs {metrics["slippage_total"]:,.0f}')
        r3[2].metric('Brokerage', f'Rs {metrics["brokerage_total"]:,.0f}')
        r3[3].metric('Total Costs', f'Rs {metrics["total_costs"]:,.0f}')
        r3[4].metric('CAGR', f'{metrics["cagr"]:.2f}%')

        st.divider()

        # Row 4: Win/Loss detail
        r4 = st.columns(5)
        r4[0].metric('Avg Win', f'Rs {metrics["avg_win"]:,.0f}')
        r4[1].metric('Avg Loss', f'Rs {metrics["avg_loss"]:,.0f}')
        r4[2].metric('Avg Win %', f'{metrics["avg_win_pct"]:.4f}%')
        r4[3].metric('Avg Loss %', f'{metrics["avg_loss_pct"]:.4f}%')
        r4[4].metric('Max Con. Losses', f'{metrics["max_con_losses"]}')

        st.divider()

        # Rolling Win Rate
        st.markdown('### Rolling Win Rate')
        st.plotly_chart(plot_rolling_win_rate(tdf), use_container_width=True)

        # Lot performance
        st.markdown('### Performance by Lot Size')
        lot_perf = compute_lot_performance(tdf)
        if not lot_perf.empty:
            st.dataframe(lot_perf, use_container_width=True)

        # Exit breakdown
        st.markdown('### Exit Breakdown')
        ec1, ec2 = st.columns(2)
        with ec1:
            exit_df = compute_exit_breakdown(tdf)
            if not exit_df.empty:
                st.dataframe(exit_df, hide_index=True, use_container_width=True)
        with ec2:
            st.plotly_chart(plot_exit_reasons_pie(tdf), use_container_width=True)

        # Monthly breakdown
        st.markdown('### Monthly Breakdown')
        monthly = compute_monthly_breakdown(tdf)
        if not monthly.empty:
            st.dataframe(monthly, use_container_width=True)
    else:
        st.warning('No trades generated. Adjust parameters in the sidebar.')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: TRADE ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[2]:
    if not tdf.empty:
        st.markdown('### Trade Log')

        # Download button
        csv_data = tdf.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Download Trades as CSV',
            data=csv_data,
            file_name='trades.csv',
            mime='text/csv',
        )

        display_cols = [
            'trade_num', 'entry_date', 'entry_time', 'exit_time',
            'lots', 'entry_price', 'exit_price', 'bars_held',
            'pnl_pct', 'gross_pnl', 'slippage', 'net_pnl',
            'exit_reason', 'regime', 'signal',
        ]
        avail_cols = [c for c in display_cols if c in tdf.columns]
        st.dataframe(
            tdf[avail_cols].style.applymap(
                lambda v: 'color: #34D399' if isinstance(v, (int, float)) and v > 0
                else ('color: #F87171' if isinstance(v, (int, float)) and v < 0 else ''),
                subset=[c for c in ['net_pnl', 'gross_pnl', 'pnl_pct'] if c in avail_cols],
            ),
            use_container_width=True,
            height=500,
        )

        st.divider()

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_per_trade_pnl(tdf), use_container_width=True)
        with c2:
            st.plotly_chart(plot_win_loss_pie(tdf), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_pnl_distribution(tdf), use_container_width=True)
        with c4:
            st.plotly_chart(plot_hold_distribution(tdf), use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(plot_pnl_by_weekday(tdf), use_container_width=True)
        with c6:
            st.plotly_chart(plot_daily_trade_count(tdf), use_container_width=True)

        # Trade scatter
        st.plotly_chart(plot_trade_scatter(tdf), use_container_width=True)

        # Hourly performance
        st.markdown('### Hourly Performance')
        st.plotly_chart(plot_hourly_pnl(tdf), use_container_width=True)

        # Position sizing
        st.markdown('### Position Size vs P&L')
        st.plotly_chart(plot_position_vs_pnl(tdf), use_container_width=True)

        # Entry Quality
        eq = compute_entry_quality(tdf)
        if eq:
            st.markdown('### Entry Quality Analysis')
            q1, q2 = st.columns(2)
            with q1:
                st.markdown('**Winners**')
                st.metric('Avg ML Signal', f'{eq["winners_signal"]:.4f}')
                st.metric('Avg Bars Held', f'{eq["winners_bars_held"]:.1f}')
            with q2:
                st.markdown('**Losers**')
                st.metric('Avg ML Signal', f'{eq["losers_signal"]:.4f}')
                st.metric('Avg Bars Held', f'{eq["losers_bars_held"]:.1f}')
    else:
        st.warning('No trades to display.')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: DAY REPLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[3]:
    if not tdf.empty:
        st.markdown('### Day Replay')
        dates = get_available_dates(tdf)
        if dates:
            selected_date = st.selectbox('Select Trading Day', dates, index=len(dates) - 1)
            day_data = get_day_data(df_raw, tdf, selected_date)
            day_summary = day_data.get('summary', {})

            if day_summary:
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric('Open', f'Rs {day_summary["spot_open"]:,.2f}')
                s2.metric('Close', f'Rs {day_summary["spot_close"]:,.2f}')
                s3.metric('Range', f'Rs {day_summary["day_range"]:,.2f}')
                s4.metric('Trades', f'{day_summary["num_trades"]}')
                s5.metric('Day P&L', f'Rs {day_summary["day_pnl"]:,.2f}')

                st.plotly_chart(plot_day_replay(day_data), use_container_width=True)

                day_trades = day_data.get('day_trades', pd.DataFrame())
                if not day_trades.empty:
                    st.markdown('#### Trades on this day')
                    show_cols = [c for c in ['entry_time', 'exit_time', 'lots',
                                              'entry_price', 'exit_price', 'pnl_pct',
                                              'net_pnl', 'exit_reason', 'signal'] if c in day_trades.columns]
                    st.dataframe(day_trades[show_cols], hide_index=True, use_container_width=True)
        else:
            st.info('No trading days available.')
    else:
        st.warning('No trades to display.')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5: REGIME ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[4]:
    if not tdf.empty:
        st.markdown('### Market Regime Analysis')

        # Regime distribution from features
        if not df_feat.empty:
            st.markdown('#### Regime Distribution (Full Dataset)')
            regime_dist = get_regime_distribution(df_feat)
            if not regime_dist.empty:
                st.dataframe(regime_dist, hide_index=True, use_container_width=True)

        # Regime P&L bar chart
        st.markdown('#### P&L by Market Regime')
        st.plotly_chart(plot_regime_pnl(tdf), use_container_width=True)

        # Regime trade performance table
        st.markdown('#### Regime Trade Performance')
        regime_perf = compute_regime_performance(tdf, REGIME_MAP)
        if not regime_perf.empty:
            st.dataframe(regime_perf, use_container_width=True)
            st.plotly_chart(plot_regime_performance(regime_perf, 'Market Regime Performance'),
                            use_container_width=True)

        st.divider()

        # Regime transitions
        if not df_feat.empty:
            st.markdown('#### Regime Transition Matrix')
            st.plotly_chart(plot_regime_transitions(df_feat), use_container_width=True)

        st.divider()

        # Additional trend/vol classification
        st.markdown('#### Trend Regime')
        tdf_regime = classify_vol_trend_regimes(df_raw, tdf)
        if 'trend_regime' in tdf_regime.columns:
            trend_perf = get_regime_performance(tdf_regime, 'trend_regime')
            if not trend_perf.empty:
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.dataframe(trend_perf, use_container_width=True)
                with tc2:
                    st.plotly_chart(plot_regime_performance(trend_perf, 'Trend Regime'), use_container_width=True)

        st.markdown('#### Volatility Regime')
        if 'vol_regime_label' in tdf_regime.columns:
            vol_perf = get_regime_performance(tdf_regime, 'vol_regime_label')
            if not vol_perf.empty:
                vc1, vc2 = st.columns(2)
                with vc1:
                    st.dataframe(vol_perf, use_container_width=True)
                with vc2:
                    st.plotly_chart(plot_regime_performance(vol_perf, 'Volatility Regime'), use_container_width=True)
    else:
        st.warning('No trades to display.')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6: ML ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[5]:
    if not tdf.empty:
        st.markdown('### ML Analytics')

        # Walk-Forward fold accuracy
        if fold_results:
            st.markdown('#### Walk-Forward Out-of-Sample Accuracy')
            st.plotly_chart(plot_fold_accuracy(fold_results), use_container_width=True)

            # Fold details table
            fold_df = pd.DataFrame(fold_results)
            st.dataframe(fold_df, hide_index=True, use_container_width=True)

        st.divider()

        # Feature importance
        if models:
            st.markdown('#### Feature Importance (Final Model)')
            final_model = models[-1]
            feat_imp = pd.Series(final_model.feature_importances_, index=feature_cols)
            st.plotly_chart(plot_feature_importance(feat_imp, top_n=15), use_container_width=True)

        st.divider()

        # Signal distribution
        if not df_feat.empty and 'ml_signal' in df_feat.columns:
            st.markdown('#### ML Signal Distribution')
            st.plotly_chart(plot_signal_distribution(df_feat, cfg['MIN_SIGNAL_PROB']), use_container_width=True)

        st.divider()

        # Signal vs P&L analysis
        st.markdown('#### Signal Quality Analysis')
        vol_analysis = compute_vol_change_analysis(tdf)
        if vol_analysis:
            va1, va2, va3 = st.columns(3)
            va1.metric('Avg Signal', f'{vol_analysis["avg_signal"]:.4f}')
            va2.metric('Winners Avg Signal', f'{vol_analysis["winners_avg_signal"]:.4f}')
            va3.metric('Signal-P&L Corr', f'{vol_analysis["signal_pnl_corr"]:.3f}')
    else:
        st.warning('No trades to display.')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 7: MONTE CARLO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[6]:
    if not tdf.empty:
        st.markdown('### Monte Carlo Simulation')
        n_sims = st.slider('Number of Simulations', 100, 5000, 1000, 100, key='mc_sims')
        mc = run_monte_carlo(tdf['net_pnl'], cfg['INITIAL_CAPITAL'], n_sims)

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric('Median Final', f'Rs {mc["median_final"]:,.0f}')
        mc2.metric('5th Pctl (Worst)', f'Rs {mc["p5_final"]:,.0f}')
        mc3.metric('95th Pctl (Best)', f'Rs {mc["p95_final"]:,.0f}')
        mc4.metric('Prob of Profit', f'{mc["prob_profit"]:.1f}%')
        mc5.metric('Prob of 2x Capital', f'{mc["prob_double"]:.1f}%')

        st.plotly_chart(plot_monte_carlo(mc), use_container_width=True)

        with st.expander('Detailed Statistics'):
            st.json({
                'Mean Final P&L': f'Rs {mc["mean_final"]:,.0f}',
                'Std Dev': f'Rs {mc["std_final"]:,.0f}',
                'Worst Case': f'Rs {mc["worst_case"]:,.0f}',
                'Best Case': f'Rs {mc["best_case"]:,.0f}',
                'Simulations': mc['n_simulations'],
                'Trades': mc['n_trades'],
            })
    else:
        st.warning('No trades to display.')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 8: ROLLING SHARPE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[7]:
    if not tdf.empty:
        st.markdown('### Rolling Sharpe Ratio')
        window = st.slider('Rolling Window (days)', 10, 120, 30, 5, key='sharpe_window')

        rolling_df = compute_rolling_sharpe(tdf, window)
        if not rolling_df.empty:
            st.plotly_chart(plot_rolling_sharpe(rolling_df, window), use_container_width=True)

        st.divider()

        st.markdown('### Multi-Window Comparison')
        mw1, mw2, mw3 = st.columns(3)
        for col, w in zip([mw1, mw2, mw3], [30, 60, 90]):
            rdf = compute_rolling_sharpe(tdf, w)
            if not rdf.empty:
                avg_sharpe = rdf['rolling_sharpe'].mean()
                col.metric(f'{w}-Day Avg Sharpe', f'{avg_sharpe:.2f}')
    else:
        st.warning('No trades to display.')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 9: ROBUSTNESS TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[8]:
    if not tdf.empty:
        st.markdown('### Robustness Testing')

        # Slippage stress
        st.markdown('#### Slippage Stress Test')
        slip_df = slippage_stress_test(tdf, cfg)
        if not slip_df.empty:
            st.plotly_chart(plot_stress_test(slip_df, 'slippage_pct', 'total_net_pnl',
                                             'Net P&L vs Slippage'), use_container_width=True)
            st.dataframe(slip_df, hide_index=True, use_container_width=True)

        st.divider()

        # Brokerage stress
        st.markdown('#### Brokerage Stress Test')
        brok_df = brokerage_stress_test(tdf, cfg)
        if not brok_df.empty:
            st.plotly_chart(plot_stress_test(brok_df, 'brokerage_flat', 'total_net_pnl',
                                             'Net P&L vs Brokerage'), use_container_width=True)
            st.dataframe(brok_df, hide_index=True, use_container_width=True)

        st.divider()

        # Parameter sensitivity
        st.markdown('#### Parameter Sensitivity — Stop vs Signal Threshold')
        sens_df = parameter_sensitivity_heatmap(tdf, cfg)
        if not sens_df.empty:
            st.plotly_chart(plot_sensitivity_heatmap(sens_df), use_container_width=True)
    else:
        st.warning('No trades to display.')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.divider()
st.markdown(
    '<div style="text-align:center;color:#64748B;padding:16px 0;font-size:0.85rem;">'
    'Regime-Switch ML Bot  ·  Walk-Forward RandomForest  ·  4-Regime Classification  ·  '
    'Built for institutional quant research'
    '</div>',
    unsafe_allow_html=True,
)
