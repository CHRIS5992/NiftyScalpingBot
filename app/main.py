"""
main.py — NIFTY ATM Straddle Scalping Bot v4.0
Production-grade Streamlit quant research dashboard.
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

from app.config import get_config_dict, FEATURES
from app.utils import inject_custom_css
from app.data_loader import load_and_clean, get_data_summary
from app.strategy import add_features, train_ml_models, load_cached_ml, run_backtest
from app.metrics import (
    compute_metrics, compute_monthly_breakdown, compute_monthly_heatmap_data,
    compute_exit_breakdown, compute_lot_performance, compute_entry_quality,
    compute_weekday_pnl,
)
from app.plots import (
    plot_cumulative_pnl, plot_per_trade_pnl, plot_win_loss_pie,
    plot_monthly_heatmap, plot_iv_vs_pnl, plot_hold_duration,
    plot_ml_signal_dist, plot_exit_reasons, plot_equity_curve,
    plot_pnl_distribution, plot_pnl_by_weekday, plot_day_replay,
    plot_iv_percentile, plot_monte_carlo, plot_rolling_sharpe,
    plot_iv_change, plot_feature_importance, plot_stress_test,
    plot_regime_performance, plot_holding_distribution,
    plot_daily_trade_count, plot_sensitivity_heatmap,
)
from app.monte_carlo import run_monte_carlo
from app.regime_analysis import classify_regimes, get_regime_performance
from app.trade_replay import get_day_data, get_available_dates
from app.iv_analytics import (
    compute_iv_percentile_series, compute_iv_vs_returns,
    classify_iv_regime, get_iv_regime_performance, compute_iv_change_analysis,
)
from app.robustness import (
    slippage_stress_test, brokerage_stress_test,
    compute_rolling_sharpe, parameter_sensitivity_heatmap,
)

CACHE_PATH = ROOT / 'cache' / 'ml_cache.pkl'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title='NIFTY Straddle Bot v4.0',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded',
)
st.markdown(inject_custom_css(), unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR — FULL CONTROL PANEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown('<h2 style="color:#FFD700;">⚡ Control Panel</h2>', unsafe_allow_html=True)
    st.caption('Adjust parameters → backtest re-runs live')
    st.divider()

    # ── Capital & Position Sizing ──
    st.markdown('#### 💰 Capital & Sizing')
    initial_capital = st.number_input('Initial Capital (₹)', 100_000, 50_000_000, 1_000_000, step=100_000)
    lot_size = st.number_input('Lot Size', 1, 500, 65, step=1)
    max_lots = st.slider('Max Lots per Trade', 1, 10, 3)

    st.divider()

    # ── Strategy Parameters ──
    st.markdown('#### 🎯 Strategy Parameters')
    target_pct = st.slider('Target %', 0.02, 0.30, 0.14, 0.01, format='%.2f')
    stop_pct = st.slider('Stop Loss %', 0.02, 0.20, 0.07, 0.01, format='%.2f')
    expiry_stop = st.slider('Expiry Stop %', 0.02, 0.15, 0.05, 0.01, format='%.2f')
    max_hold_bars = st.slider('Max Hold (bars)', 3, 20, 9)
    max_trades_day = st.slider('Max Trades / Day', 1, 10, 3)
    cooldown = st.slider('Cooldown (bars)', 1, 10, 4)
    loss_streak = st.slider('Loss Streak Limit', 1, 10, 3)

    st.divider()

    # ── ML Parameters ──
    st.markdown('#### 🧠 ML Parameters')
    ml_threshold = st.slider('ML Threshold', 0.30, 0.70, 0.46, 0.01, format='%.2f')
    iv_zscore_boost = st.slider('IV Z-Score Boost', 0.0, 2.0, 0.5, 0.1, format='%.1f')
    iv_min = st.slider('IV Minimum', 1.0, 30.0, 8.5, 0.5, format='%.1f')
    iv_rank_min = st.slider('IV Rank Min', 0.0, 0.5, 0.15, 0.05, format='%.2f')
    iv_rank_max = st.slider('IV Rank Max', 0.5, 1.0, 0.85, 0.05, format='%.2f')

    st.divider()

    # ── Cost Parameters ──
    st.markdown('#### 💸 Costs')
    slippage_pct = st.slider('Slippage %', 0.0, 0.01, 0.001, 0.0005, format='%.4f')
    brokerage = st.number_input('Brokerage (₹/trade)', 0, 500, 40, step=10)

    st.divider()

    # ── Session ──
    st.markdown('#### 🕐 Session')
    session_start = st.slider('Session Start Bar', 0, 10, 3)
    session_end = st.slider('Session End Bar', 50, 70, 62)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILD CONFIG FROM SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cfg = get_config_dict(
    INITIAL_CAPITAL=initial_capital,
    LOT_SIZE=lot_size,
    MAX_LOTS_PER_TRADE=max_lots,
    TARGET_PCT=target_pct,
    STOP_PCT=stop_pct,
    EXPIRY_STOP=expiry_stop,
    MAX_HOLD_BARS=max_hold_bars,
    MAX_TRADES_PER_DAY=max_trades_day,
    COOLDOWN_BARS=cooldown,
    LOSS_STREAK_LIMIT=loss_streak,
    ML_THRESHOLD=ml_threshold,
    IV_ZSCORE_BOOST=iv_zscore_boost,
    IV_MIN=iv_min,
    IV_RANK_MIN=iv_rank_min,
    IV_RANK_MAX=iv_rank_max,
    SLIPPAGE_PCT=slippage_pct,
    BROKERAGE_FLAT=brokerage,
    SESSION_START_BAR=session_start,
    SESSION_END_BAR=session_end,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING & PIPELINE (cache-first!)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(show_spinner=False)
def load_data():
    """Load and clean data. Cached to avoid re-reading CSV."""
    csv_path = ROOT / 'FINAL_NIFTY_MASTER_ATM_5min.csv'
    if not csv_path.exists():
        for alt in [ROOT / 'data' / 'FINAL_NIFTY_MASTER_ATM_5min.csv']:
            if alt.exists():
                csv_path = alt
                break
    return load_and_clean(csv_path)


@st.cache_data(show_spinner=False)
def run_feature_engineering(_df_raw, rs, rm, rl, roll_rank, future_bars):
    """Feature engineering. Cached on window parameters."""
    feat_cfg = {'RS': rs, 'RM': rm, 'RL': rl, 'ROLL_RANK': roll_rank, 'FUTURE_BARS': future_bars}
    df_feat = add_features(_df_raw, feat_cfg)
    df_feat = df_feat.dropna(subset=['trade_label']).reset_index(drop=True)
    return df_feat


@st.cache_data(show_spinner=False)
def get_ml_data(_df_feat, cache_exists):
    """Load ML data from cache or train from scratch."""
    if cache_exists:
        return load_cached_ml(_df_feat, CACHE_PATH)
    else:
        cfg_full = get_config_dict()
        df_c, rf, gb, scaler, col_means, ml_stats = train_ml_models(_df_feat, cfg_full)
        return df_c, ml_stats


# ── Execute Pipeline with Progress ──
pipeline_ok = False
tdf = pd.DataFrame()
equity_curve = []
lot_usage = {}
metrics = {}
ml_stats = {}
data_summary = {}

has_cache = CACHE_PATH.exists()
progress_bar = st.progress(0, text='⏳ Loading data...')
try:
    df_raw = load_data()
    data_summary = get_data_summary(df_raw)
    progress_bar.progress(20, text=f'✅ Data loaded — {data_summary["total_rows"]:,} rows  |  Engineering features...')

    df_feat = run_feature_engineering(
        df_raw, cfg['RS'], cfg['RM'], cfg['RL'], cfg['ROLL_RANK'], cfg['FUTURE_BARS']
    )

    if has_cache:
        progress_bar.progress(40, text='✅ Features built  |  Loading ML from cache...')
    else:
        progress_bar.progress(40, text='✅ Features built  |  Training ML (first run, ~3 min)...')

    df_c, ml_stats = get_ml_data(df_feat, has_cache)
    progress_bar.progress(70, text=f'✅ ML ready (AUC: {ml_stats["ens_auc"]:.4f})  |  Running backtest...')

    tdf, equity_curve, lot_usage = run_backtest(df_c, cfg)
    metrics = compute_metrics(tdf, cfg)
    pipeline_ok = True

    progress_bar.progress(100, text=f'✅ Done — {metrics.get("total_trades", 0)} trades  |  Net P&L: ₹{metrics.get("net_pnl", 0):,.0f}')
except Exception as e:
    st.error(f'Pipeline error: {e}')
    import traceback
    st.code(traceback.format_exc())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div class="gradient-header">⚡ NIFTY ATM Straddle Scalping Bot v4.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ML-Enhanced Gamma / Volatility Scalping  •  RF + GBM Ensemble  •  Buy Side Only</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tabs = st.tabs([
    '📋 Overview',
    '📊 Results & Equity',
    '🔍 Trades',
    '🎬 Day Replay',
    '📉 IV Analytics',
    '🌐 Regimes',
    '🎲 Monte Carlo',
    '📐 Rolling Sharpe',
    '🧪 Robustness',
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: STRATEGY OVERVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[0]:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('#### 🎯 Strategy Description')
        st.markdown("""
        **NIFTY ATM Straddle Gamma/Volatility Scalping** — an intraday strategy that buys ATM straddles
        (both CE + PE at-the-money) when ML models predict a short-term IV spike and premium increase.

        The strategy uses a **Random Forest + Gradient Boosting ensemble** trained on 45 features including
        IV dynamics, straddle premium momentum, spot realized volatility, open interest, and volume surges.

        **Key Innovation:** Dynamic lot sizing based on ML confidence — higher confidence trades get larger positions.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('#### ⚙️ Entry Conditions')
        st.markdown(f"""
        1. ML ensemble confidence ≥ **{cfg['ML_THRESHOLD']}** (drops to 0.45 when IV z-score > {cfg['IV_ZSCORE_BOOST']})
        2. IV ≥ **{cfg['IV_MIN']}%** and IV rank ∈ [{cfg['IV_RANK_MIN']}, {cfg['IV_RANK_MAX']}]
        3. Session bar ∈ [{cfg['SESSION_START_BAR']}, {cfg['SESSION_END_BAR']}] (skip open noise + late-day)
        4. Cooldown ≥ **{cfg['COOLDOWN_BARS']}** bars after last exit
        5. Daily trades < **{cfg['MAX_TRADES_PER_DAY']}**
        6. No active loss streak ≥ **{cfg['LOSS_STREAK_LIMIT']}**
        7. Expected profit > 2× estimated costs
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('#### 🚪 Exit Rules')
        st.markdown(f"""
        1. **Target:** Straddle up **+{cfg['TARGET_PCT']*100:.0f}%** → book profit
        2. **Stop Loss:** Down **-{cfg['STOP_PCT']*100:.0f}%** (or **-{cfg['EXPIRY_STOP']*100:.0f}%** on expiry Tuesday)
        3. **Trailing Stop:** When P&L > 5%, trail at 3% from peak
        4. **Time Exit:** After **{cfg['MAX_HOLD_BARS']}** bars ({cfg['MAX_HOLD_BARS']*5} min)
        5. **Day End:** Forced close after bar 67
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('#### 📋 Parameters')
        params_data = {
            'Parameter': ['Instrument', 'Timeframe', 'Type', 'Lot Size', 'ML Models',
                          'Expiry Day', 'Capital', 'Max Risk/Trade'],
            'Value': ['NIFTY 50 ATM Straddle', '5-min bars', 'Intraday Scalping',
                      str(cfg['LOT_SIZE']), 'RF(300) + GBM(200)',
                      'Tuesday', f'₹{cfg["INITIAL_CAPITAL"]:,}', f'{cfg["MAX_CAPITAL_PER_TRADE"]*100:.0f}%'],
        }
        st.dataframe(pd.DataFrame(params_data), hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Data Summary
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('#### 📦 Data Summary')
        if data_summary:
            st.metric('Period', f'{data_summary["start_date"]} → {data_summary["end_date"]}')
            c2a, c2b = st.columns(2)
            c2a.metric('Total Rows', f'{data_summary["total_rows"]:,}')
            c2b.metric('Trading Days', f'{data_summary["trading_days"]:,}')
            st.metric('Missing Data', f'{data_summary["missing_pct"]:.2f}%')
        st.markdown('</div>', unsafe_allow_html=True)

        # ML Stats
        if ml_stats:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('#### 🧠 ML Performance')
            st.metric('RF AUC', f'{ml_stats["rf_auc"]:.4f}')
            st.metric('GBM AUC', f'{ml_stats["gb_auc"]:.4f}')
            st.metric('Ensemble AUC', f'{ml_stats["ens_auc"]:.4f}')
            st.metric('Label %', f'{ml_stats["label_pct_positive"]:.1f}% positive')
            st.markdown('</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: BACKTEST RESULTS & EQUITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[1]:
    if not tdf.empty:
        st.markdown('### 🏆 Key Metrics')

        # Row 1: Top-line metrics
        r1 = st.columns(5)
        r1[0].metric('Net P&L', f'₹{metrics["net_pnl"]:,.0f}',
                     delta=f'{metrics["roi"]:.1f}% ROI')
        r1[1].metric('CAGR', f'{metrics["cagr"]:.1f}%')
        r1[2].metric('Win Rate', f'{metrics["win_rate"]:.1f}%',
                     delta=f'{metrics["winning_trades"]}W / {metrics["losing_trades"]}L')
        r1[3].metric('Sharpe Ratio', f'{metrics["sharpe"]:.2f}')
        r1[4].metric('Max Drawdown', f'₹{metrics["max_drawdown"]:,.0f}')

        st.divider()

        # Row 2: Detailed metrics
        r2 = st.columns(5)
        r2[0].metric('Total Trades', f'{metrics["total_trades"]}')
        r2[1].metric('Profit Factor', f'{metrics["profit_factor"]:.2f}')
        r2[2].metric('Sortino Ratio', f'{metrics["sortino"]:.2f}')
        r2[3].metric('Avg Hold', f'{metrics["avg_hold_min"]:.0f} min')
        r2[4].metric('Risk/Reward', f'{metrics["risk_reward"]:.2f}x')

        st.divider()

        # ── Equity Graphs ──
        st.markdown('### 📈 Equity Curve')
        st.plotly_chart(plot_equity_curve(equity_curve, tdf, cfg, metrics), use_container_width=True)

        st.markdown('### 📉 Cumulative P&L + Drawdown')
        st.plotly_chart(plot_cumulative_pnl(tdf, metrics), use_container_width=True)

        st.markdown('### 📊 Monthly P&L Heatmap')
        piv = compute_monthly_heatmap_data(tdf)
        if not piv.empty:
            st.plotly_chart(plot_monthly_heatmap(piv), use_container_width=True)

        st.divider()

        # Row 3: P&L breakdown
        r3 = st.columns(5)
        r3[0].metric('Gross P&L', f'₹{metrics["gross_pnl"]:,.0f}')
        r3[1].metric('Slippage', f'₹{metrics["slippage_total"]:,.0f}')
        r3[2].metric('Brokerage', f'₹{metrics["brokerage_total"]:,.0f}')
        r3[3].metric('Taxes (est)', f'₹{metrics["taxes"]:,.0f}')
        r3[4].metric('Total Costs', f'₹{metrics["total_costs"]:,.0f}')

        st.divider()

        # Row 4: Win/Loss detail
        r4 = st.columns(5)
        r4[0].metric('Avg Win', f'₹{metrics["avg_win"]:,.0f}')
        r4[1].metric('Avg Loss', f'₹{metrics["avg_loss"]:,.0f}')
        r4[2].metric('Avg Win %', f'{metrics["avg_win_pct"]:.2f}%')
        r4[3].metric('Avg Loss %', f'{metrics["avg_loss_pct"]:.2f}%')
        r4[4].metric('Max Con. Losses', f'{metrics["max_con_losses"]}')

        st.divider()

        # Lot performance
        st.markdown('### 📊 Performance by Lot Size')
        lot_perf = compute_lot_performance(tdf)
        if not lot_perf.empty:
            st.dataframe(lot_perf, use_container_width=True)

        # Exit breakdown
        st.markdown('### 🚪 Exit Breakdown')
        ec1, ec2 = st.columns(2)
        with ec1:
            exit_df = compute_exit_breakdown(tdf)
            if not exit_df.empty:
                st.dataframe(exit_df, hide_index=True, use_container_width=True)
        with ec2:
            st.plotly_chart(plot_exit_reasons(tdf), use_container_width=True)

        # Monthly breakdown
        st.markdown('### 📅 Monthly Breakdown')
        monthly = compute_monthly_breakdown(tdf)
        if not monthly.empty:
            st.dataframe(monthly, use_container_width=True)

        # Feature importance
        if ml_stats and 'feat_imp' in ml_stats:
            st.markdown('### 🧠 Feature Importance')
            st.plotly_chart(plot_feature_importance(ml_stats['feat_imp']), use_container_width=True)
    else:
        st.warning('No trades generated. Adjust parameters in the sidebar.')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: TRADE ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[2]:
    if not tdf.empty:
        st.markdown('### 📋 Trade Log')
        display_cols = [
            'trade_num', 'date', 'entry_datetime', 'exit_datetime',
            'lots', 'entry_price', 'exit_price', 'entry_iv', 'exit_iv',
            'iv_change', 'bars_held', 'pnl_pct', 'gross_pnl', 'slip_cost',
            'net_pnl', 'exit_reason', 'is_expiry', 'ml_signal',
        ]
        avail_cols = [c for c in display_cols if c in tdf.columns]
        st.dataframe(
            tdf[avail_cols].style.applymap(
                lambda v: 'color: #00E676' if isinstance(v, (int, float)) and v > 0
                else ('color: #FF5252' if isinstance(v, (int, float)) and v < 0 else ''),
                subset=[c for c in ['net_pnl', 'gross_pnl', 'pnl_pct', 'iv_change'] if c in avail_cols],
            ),
            use_container_width=True,
            height=500,
        )

        st.divider()

        # Charts row
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_per_trade_pnl(tdf), use_container_width=True)
        with c2:
            st.plotly_chart(plot_win_loss_pie(tdf), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_pnl_distribution(tdf), use_container_width=True)
        with c4:
            st.plotly_chart(plot_holding_distribution(tdf), use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(plot_pnl_by_weekday(tdf), use_container_width=True)
        with c6:
            st.plotly_chart(plot_daily_trade_count(tdf), use_container_width=True)

        st.plotly_chart(plot_hold_duration(tdf, cfg['MAX_HOLD_BARS']), use_container_width=True)
        st.plotly_chart(plot_ml_signal_dist(tdf, cfg['ML_THRESHOLD']), use_container_width=True)

        # Entry Quality
        eq = compute_entry_quality(tdf)
        if eq:
            st.markdown('### 🔬 Entry Quality Analysis')
            q1, q2 = st.columns(2)
            with q1:
                st.markdown('**Winners**')
                st.metric('Avg ML Signal', f'{eq["winners_ml_signal"]:.4f}')
                st.metric('Avg Entry IV', f'{eq["winners_entry_iv"]:.2f}%')
                st.metric('Avg Bars Held', f'{eq["winners_bars_held"]:.1f}')
            with q2:
                st.markdown('**Losers**')
                st.metric('Avg ML Signal', f'{eq["losers_ml_signal"]:.4f}')
                st.metric('Avg Entry IV', f'{eq["losers_entry_iv"]:.2f}%')
                st.metric('Avg Bars Held', f'{eq["losers_bars_held"]:.1f}')
    else:
        st.warning('No trades to display.')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: DAY REPLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[3]:
    if not tdf.empty:
        st.markdown('### 🎬 Day Replay')
        dates = get_available_dates(tdf)
        if dates:
            selected_date = st.selectbox('Select Trading Day', dates, index=len(dates) - 1)
            day_data = get_day_data(df_raw, tdf, selected_date)
            summary = day_data.get('summary', {})

            if summary:
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric('Spot Open', f'₹{summary["spot_open"]:,.2f}')
                s2.metric('Spot Close', f'₹{summary["spot_close"]:,.2f}')
                s3.metric('Day Range', f'₹{summary["day_range"]:,.2f}')
                s4.metric('Trades', f'{summary["num_trades"]}')
                s5.metric('Day P&L', f'₹{summary["day_pnl"]:,.2f}')

                st.plotly_chart(plot_day_replay(day_data), use_container_width=True)

                # Show trades for the day
                day_trades = day_data.get('day_trades', pd.DataFrame())
                if not day_trades.empty:
                    st.markdown('#### Trades on this day')
                    st.dataframe(day_trades[
                        [c for c in ['entry_datetime', 'exit_datetime', 'lots',
                         'entry_price', 'exit_price', 'pnl_pct', 'net_pnl',
                         'exit_reason', 'ml_signal'] if c in day_trades.columns]
                    ], hide_index=True, use_container_width=True)
        else:
            st.info('No trading days available.')
    else:
        st.warning('No trades to display.')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5: IV ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[4]:
    if not tdf.empty:
        st.markdown('### 📉 IV Analytics')

        # IV percentile over time
        iv_pct_data = compute_iv_percentile_series(df_raw, cfg['ROLL_RANK'])
        st.plotly_chart(plot_iv_percentile(iv_pct_data), use_container_width=True)

        # IV vs Returns scatter
        st.plotly_chart(plot_iv_vs_pnl(tdf), use_container_width=True)

        # IV change per trade
        st.plotly_chart(plot_iv_change(tdf), use_container_width=True)

        # IV Regime classification
        st.markdown('### IV Regime Performance')
        tdf_iv = classify_iv_regime(tdf)
        iv_regime_perf = get_iv_regime_performance(tdf_iv)
        if not iv_regime_perf.empty:
            st.dataframe(iv_regime_perf, use_container_width=True)
            st.plotly_chart(plot_regime_performance(iv_regime_perf, 'IV Regime Performance'),
                           use_container_width=True)

        # IV change analysis
        iv_analysis = compute_iv_change_analysis(tdf)
        if iv_analysis:
            st.markdown('### IV Change Statistics')
            ic1, ic2, ic3 = st.columns(3)
            ic1.metric('Avg IV Change', f'{iv_analysis["avg_iv_change"]:.2f}%')
            ic2.metric('IV-P&L Correlation', f'{iv_analysis["iv_pnl_corr"]:.3f}')
            ic3.metric('ML-IV Correlation', f'{iv_analysis["ml_iv_corr"]:.3f}')
    else:
        st.warning('No trades to display.')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6: REGIME ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[5]:
    if not tdf.empty:
        st.markdown('### 🌐 Market Regime Analysis')
        tdf_regime = classify_regimes(df_raw, tdf)

        # Trend regime
        st.markdown('#### 📈 Trend Regime')
        trend_perf = get_regime_performance(tdf_regime, 'trend_regime')
        if not trend_perf.empty:
            tc1, tc2 = st.columns(2)
            with tc1:
                st.dataframe(trend_perf, use_container_width=True)
            with tc2:
                st.plotly_chart(plot_regime_performance(trend_perf, 'Trend Regime Performance'),
                               use_container_width=True)

        st.divider()

        # Volatility regime
        st.markdown('#### 📊 Volatility Regime')
        vol_perf = get_regime_performance(tdf_regime, 'vol_regime')
        if not vol_perf.empty:
            vc1, vc2 = st.columns(2)
            with vc1:
                st.dataframe(vol_perf, use_container_width=True)
            with vc2:
                st.plotly_chart(plot_regime_performance(vol_perf, 'Volatility Regime Performance'),
                               use_container_width=True)
    else:
        st.warning('No trades to display.')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 7: MONTE CARLO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[6]:
    if not tdf.empty:
        st.markdown('### 🎲 Monte Carlo Simulation')
        n_sims = st.slider('Number of Simulations', 100, 5000, 1000, 100, key='mc_sims')
        mc = run_monte_carlo(tdf['net_pnl'], cfg['INITIAL_CAPITAL'], n_sims)

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric('Median Final', f'₹{mc["median_final"]:,.0f}')
        mc2.metric('5th Pctl (Worst)', f'₹{mc["p5_final"]:,.0f}')
        mc3.metric('95th Pctl (Best)', f'₹{mc["p95_final"]:,.0f}')
        mc4.metric('Prob of Profit', f'{mc["prob_profit"]:.1f}%')
        mc5.metric('Prob of 2x', f'{mc["prob_double"]:.1f}%')

        st.plotly_chart(plot_monte_carlo(mc), use_container_width=True)

        with st.expander('📊 Detailed Statistics'):
            st.json({
                'Mean Final': f'₹{mc["mean_final"]:,.0f}',
                'Std Dev': f'₹{mc["std_final"]:,.0f}',
                'Worst Case': f'₹{mc["worst_case"]:,.0f}',
                'Best Case': f'₹{mc["best_case"]:,.0f}',
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
        st.markdown('### 📐 Rolling Sharpe Ratio')
        window = st.slider('Rolling Window (days)', 10, 120, 30, 5, key='sharpe_window')

        rolling_df = compute_rolling_sharpe(tdf, window)
        if not rolling_df.empty:
            st.plotly_chart(plot_rolling_sharpe(rolling_df, window), use_container_width=True)

        st.divider()

        # Multi-window comparison
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
        st.markdown('### 🧪 Robustness Testing')

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
        st.markdown('#### Parameter Sensitivity — Stop vs Target')
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
    '<div style="text-align:center;color:#8b949e;padding:16px 0;">'
    '⚡ NIFTY ATM Straddle Bot v4.0  •  ML-Enhanced Gamma Scalping  •  '
    'Built for institutional quant research'
    '</div>',
    unsafe_allow_html=True,
)
