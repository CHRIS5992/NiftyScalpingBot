"""
strategy.py — Feature engineering, ML model training, and backtest engine.
Exact reproduction of notebook Cells 3, 4, 6.

NO UI LOGIC. Pure computation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from .utils import get_lot_size


# ============================================================
# CELL 3 — FEATURE ENGINEERING
# ============================================================

def add_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add all 45 features + target labels. Exact Cell 3 logic.

    Args:
        df: Raw cleaned DataFrame from data_loader.
        cfg: Config dict with RS, RM, RL, ROLL_RANK, FUTURE_BARS keys.

    Returns:
        DataFrame with all features and target labels added.
    """
    df = df.copy()
    RS = cfg['RS']
    RM = cfg['RM']
    RL = cfg['RL']
    ROLL_RANK = cfg['ROLL_RANK']
    FUTURE_BARS = cfg['FUTURE_BARS']

    # IV levels, lags, momentum
    df['iv_lag1']       = df['iv'].shift(1)
    df['iv_lag2']       = df['iv'].shift(2)
    df['iv_lag3']       = df['iv'].shift(3)
    df['iv_lag5']       = df['iv'].shift(RS)
    df['iv_ma_s']       = df['iv'].rolling(RS, min_periods=1).mean()
    df['iv_ma_m']       = df['iv'].rolling(RM, min_periods=1).mean()
    df['iv_ma_l']       = df['iv'].rolling(RL, min_periods=1).mean()
    df['iv_std_s']      = df['iv'].rolling(RS, min_periods=2).std().bfill().ffill()
    df['iv_std_m']      = df['iv'].rolling(RM, min_periods=2).std().bfill().ffill()
    df['iv_std_l']      = df['iv'].rolling(RL, min_periods=2).std().bfill().ffill()
    df['iv_z']          = (df['iv'] - df['iv_ma_m']) / (df['iv_std_m'] + 1e-8)
    df['iv_z_burst']    = df['iv_z'].rolling(RS, min_periods=1).max()
    df['iv_rank']       = df['iv'].rolling(ROLL_RANK, min_periods=20).rank(pct=True)
    df['iv_pct1']       = df['iv'].pct_change(1)
    df['iv_pct_s']      = df['iv'].pct_change(RS)
    df['iv_diff1']      = df['iv'].diff(1)
    df['iv_diff_s']     = df['iv'].diff(RS)
    df['iv_ma_cross']   = (df['iv_ma_s'] > df['iv_ma_m']).astype(int)
    df['iv_accel']      = df['iv_diff1'].diff(1)

    # Straddle premium features
    df['prem_ret1']     = df['Straddle_Price'].pct_change(1)
    df['prem_ret_s']    = df['Straddle_Price'].pct_change(RS)
    df['prem_ma_m']     = df['Straddle_Price'].rolling(RM, min_periods=1).mean()
    df['prem_ma_ratio'] = df['Straddle_Price'] / (df['prem_ma_m'] + 1e-8)
    df['prem_std_s']    = df['Straddle_Price'].rolling(RS, min_periods=2).std().bfill().ffill()

    # Spot / realized volatility
    df['spot_ret1']     = df['spot'].pct_change(1)
    df['spot_ret_s']    = df['spot'].pct_change(RS)
    df['spot_ret_m']    = df['spot'].pct_change(RM)
    df['spot_rv_s']     = df['spot'].pct_change(1).rolling(RS, min_periods=2).std() * np.sqrt(252)
    df['spot_rv_m']     = df['spot'].pct_change(1).rolling(RM, min_periods=2).std() * np.sqrt(252)
    df['rv_iv_ratio']   = df['spot_rv_s'] / (df['iv'] / 100 + 1e-8)
    df['gamma_proxy']   = df['Straddle_Price'] / df['spot'] * 100

    # Open Interest & Volume
    df['total_oi']      = df['CE_oi'] + df['PE_oi']
    df['oi_ratio']      = df['CE_oi'] / (df['PE_oi'] + 1e-8)
    df['oi_chg1']       = df['total_oi'].pct_change(1)
    df['oi_chg_s']      = df['total_oi'].pct_change(RS)
    df['total_vol']     = df['CE_volume'] + df['PE_volume']
    df['vol_ma_s']      = df['total_vol'].rolling(RS, min_periods=1).mean()
    df['vol_surge']     = df['total_vol'] / (df['vol_ma_s'] + 1e-8)
    df['cp_vol_ratio']  = df['CE_volume'] / (df['PE_volume'] + 1e-8)
    df['pcp']           = (df['CE_close'] - df['PE_close']) / (df['Straddle_Price'] + 1e-8)

    # Session & time features — EXPIRY = TUESDAY (dow=1) CORRECTED
    df['hour']          = df['datetime'].dt.hour
    df['minute']        = df['datetime'].dt.minute
    df['dow']           = df['datetime'].dt.dayofweek
    df['is_expiry']     = (df['dow'] == 1).astype(int)  # Tuesday
    df['time_from_open'] = (df['hour'] * 60 + df['minute']) - (9 * 60 + 15)
    df['bar_num']       = df.groupby('date').cumcount()
    df['intraday_range'] = (
        df.groupby('date')['spot'].transform('cummax')
        - df.groupby('date')['spot'].transform('cummin')
    )

    # Target labels (forward-shift)
    df['iv_up']         = (df['iv'].shift(-FUTURE_BARS) > df['iv'] * 1.003).astype(int)
    df['prem_up']       = (df['Straddle_Price'].shift(-FUTURE_BARS)
                           > df['Straddle_Price'] * 1.006).astype(int)
    df['trade_label']   = ((df['iv_up'] == 1) & (df['prem_up'] == 1)).astype(int)

    return df


# ============================================================
# CELL 4 — ML MODEL TRAINING
# ============================================================

def train_ml_models(df_feat: pd.DataFrame, cfg: dict) -> tuple:
    """Train RF + GBM ensemble. Exact Cell 4 logic.

    Args:
        df_feat: DataFrame with features and trade_label (after add_features + dropna).
        cfg: Config dict.

    Returns:
        (df_c, rf_model, gb_model, scaler, col_means, ml_stats)
        where df_c has 'ml_signal' column added.
    """
    FEATURES = cfg['FEATURES']
    TRAIN_RATIO = cfg['TRAIN_RATIO']
    RANDOM_STATE = cfg['RANDOM_STATE']

    df_c = df_feat.copy()
    for col in FEATURES:
        df_c[col] = df_c[col].replace([np.inf, -np.inf], np.nan)

    split = int(len(df_c) * TRAIN_RATIO)
    dtr = df_c.iloc[:split].dropna(subset=FEATURES)
    dte = df_c.iloc[split:].dropna(subset=FEATURES)

    Xtr_r = dtr[FEATURES].values.copy()
    Xte_r = dte[FEATURES].values.copy()
    ytr   = dtr['trade_label'].values
    yte   = dte['trade_label'].values

    # Fill NaN with TRAIN column means only
    col_means = np.nanmean(Xtr_r, axis=0)
    nan_mask_tr = np.isnan(Xtr_r)
    if nan_mask_tr.any():
        Xtr_r[nan_mask_tr] = np.take(col_means, np.where(nan_mask_tr)[1])
    nan_mask_te = np.isnan(Xte_r)
    if nan_mask_te.any():
        Xte_r[nan_mask_te] = np.take(col_means, np.where(nan_mask_te)[1])

    # Fit scaler on TRAIN only
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_r)
    Xte = scaler.transform(Xte_r)

    # Model 1: Random Forest
    rf = RandomForestClassifier(
        n_estimators=cfg['RF_TREES'],
        max_depth=cfg['RF_MAX_DEPTH'],
        min_samples_leaf=cfg['RF_MIN_SAMPLES_LEAF'],
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(Xtr, ytr)
    rf_p   = rf.predict_proba(Xte)[:, 1]
    rf_auc = roc_auc_score(yte, rf_p)

    # Model 2: Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=cfg['GB_TREES'],
        max_depth=cfg['GB_MAX_DEPTH'],
        learning_rate=cfg['GB_LEARNING_RATE'],
        subsample=cfg['GB_SUBSAMPLE'],
        random_state=RANDOM_STATE,
    )
    gb.fit(Xtr, ytr)
    gb_p   = gb.predict_proba(Xte)[:, 1]
    gb_auc = roc_auc_score(yte, gb_p)

    # Equal weighted ensemble
    rf_w = cfg['RF_WEIGHT']
    gb_w = cfg['GB_WEIGHT']
    ens_p   = rf_w * rf_p + gb_w * gb_p
    ens_auc = roc_auc_score(yte, ens_p)

    # Generate signal for ALL rows
    Xf_r = df_c[FEATURES].values.copy()
    nan_mask_f = np.isnan(Xf_r)
    if nan_mask_f.any():
        Xf_r[nan_mask_f] = np.take(col_means, np.where(nan_mask_f)[1])
    Xf = scaler.transform(Xf_r)
    df_c['ml_signal'] = rf_w * rf.predict_proba(Xf)[:, 1] + gb_w * gb.predict_proba(Xf)[:, 1]

    # Feature importance
    feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    ml_stats = {
        'rf_auc': rf_auc,
        'gb_auc': gb_auc,
        'ens_auc': ens_auc,
        'train_rows': len(dtr),
        'test_rows': len(dte),
        'train_start': dtr['date'].min(),
        'train_end': dtr['date'].max(),
        'test_start': dte['date'].min(),
        'test_end': dte['date'].max(),
        'feat_imp': feat_imp,
        'ml_signal_min': df_c['ml_signal'].min(),
        'ml_signal_mean': df_c['ml_signal'].mean(),
        'ml_signal_max': df_c['ml_signal'].max(),
        'label_pct_positive': df_feat['trade_label'].mean() * 100,
    }

    return df_c, rf, gb, scaler, col_means, ml_stats


def load_cached_ml(df_feat: pd.DataFrame, cache_path) -> tuple:
    """Load pre-computed ml_signal from disk cache.

    Returns (df_c, ml_stats) — same as train_ml_models but instant.
    """
    import pickle
    from pathlib import Path

    cache_path = Path(cache_path)
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    df_c = df_feat.copy()
    for col in df_c.columns:
        if df_c[col].dtype in ['float64', 'float32']:
            df_c[col] = df_c[col].replace([np.inf, -np.inf], np.nan)

    # Apply cached ml_signal
    df_c['ml_signal'] = cache['ml_signal']

    # Reconstruct ml_stats
    feat_imp = pd.Series(
        cache['feat_imp_values'],
        index=cache['feat_imp_index'],
    )
    ml_stats = cache['ml_stats'].copy()
    ml_stats['feat_imp'] = feat_imp

    return df_c, ml_stats


# ============================================================
# CELL 6 — BACKTEST ENGINE (v4.1 - Correct Expiry Day)
# ============================================================

def run_backtest(df_c: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list, dict]:
    """Run the full backtest. Exact Cell 6 logic preserved.

    OPTIMIZED: Pre-extracts columns as numpy arrays to avoid
    iterrows() overhead. Produces identical results.
    """
    np.random.seed(cfg['RANDOM_STATE'])

    ML_LOT_MAPPING     = cfg['ML_LOT_MAPPING']
    LOT_SIZE            = cfg['LOT_SIZE']
    BASE_LOTS           = cfg['BASE_LOTS']
    INITIAL_CAPITAL     = cfg['INITIAL_CAPITAL']
    ML_THRESHOLD        = cfg['ML_THRESHOLD']
    IV_ZSCORE_BOOST     = cfg['IV_ZSCORE_BOOST']
    IV_MIN              = cfg['IV_MIN']
    IV_RANK_MIN         = cfg['IV_RANK_MIN']
    IV_RANK_MAX         = cfg['IV_RANK_MAX']
    TARGET_PCT          = cfg['TARGET_PCT']
    STOP_PCT            = cfg['STOP_PCT']
    EXPIRY_STOP         = cfg['EXPIRY_STOP']
    MAX_HOLD_BARS       = cfg['MAX_HOLD_BARS']
    MAX_TRADES_PER_DAY  = cfg['MAX_TRADES_PER_DAY']
    COOLDOWN_BARS       = cfg['COOLDOWN_BARS']
    SESSION_START_BAR   = cfg['SESSION_START_BAR']
    SESSION_END_BAR     = cfg['SESSION_END_BAR']
    LOSS_STREAK_LIMIT   = cfg['LOSS_STREAK_LIMIT']
    SLIPPAGE_PCT        = cfg['SLIPPAGE_PCT']
    BROKERAGE_FLAT      = cfg['BROKERAGE_FLAT']
    MAX_CAPITAL_PER_TRADE = cfg['MAX_CAPITAL_PER_TRADE']

    # ── Pre-extract all columns as arrays (the key optimization) ──
    n_rows = len(df_c)
    arr_date       = df_c['date'].values          # object array of strings
    arr_bar_num    = df_c['bar_num'].values.astype(np.int32)
    arr_is_expiry  = df_c['is_expiry'].values.astype(np.int32)
    arr_price      = df_c['Straddle_Price'].values.astype(np.float64)
    arr_iv         = df_c['iv'].values.astype(np.float64)
    arr_datetime   = df_c['datetime'].values       # datetime64 array
    arr_ml_signal  = df_c['ml_signal'].values.astype(np.float64)
    arr_iv_z       = df_c['iv_z'].values.astype(np.float64)
    arr_iv_rank    = df_c['iv_rank'].values.astype(np.float64)
    arr_spot       = df_c['spot'].values.astype(np.float64)

    trades = []
    in_trade = False
    ep = eb = eiv = esp = 0.0
    edt = edate = eml = None
    current_lots = BASE_LOTS

    daily_trades = {}
    daily_last_exit = {}
    capital = float(INITIAL_CAPITAL)
    equity_curve = [(None, capital)]
    consecutive_losses = 0

    monthly_performance = {}
    lot_usage = {1: 0, 2: 0, 3: 0}

    for idx in range(n_rows):
        date      = str(arr_date[idx])
        month_key = date[:7]
        bar       = int(arr_bar_num[idx])
        expiry    = bool(arr_is_expiry[idx])
        price     = float(arr_price[idx])
        iv        = float(arr_iv[idx])
        cur_dt    = arr_datetime[idx]
        ml        = float(arr_ml_signal[idx])
        iv_z_val  = arr_iv_z[idx]
        iv_z      = 0.0 if np.isnan(iv_z_val) else float(iv_z_val)

        # ── ENTRY ──
        if not in_trade:
            if consecutive_losses >= LOSS_STREAK_LIMIT:
                consecutive_losses = max(0, consecutive_losses - 1)
                continue
            if daily_trades.get(date, 0) >= MAX_TRADES_PER_DAY:
                continue
            if bar - daily_last_exit.get(date, -99) < COOLDOWN_BARS:
                continue
            if bar < SESSION_START_BAR or bar > SESSION_END_BAR:
                continue

            iv_rank = arr_iv_rank[idx]
            if np.isnan(iv_rank):
                continue

            if month_key in monthly_performance:
                if monthly_performance[month_key] < -5000:
                    if np.random.random() < 0.5:
                        continue

            threshold = 0.45 if iv_z > IV_ZSCORE_BOOST else ML_THRESHOLD
            if bar < 10:
                threshold = min(threshold + 0.03, 0.60)

            expected_profit = price * TARGET_PCT * LOT_SIZE
            estimated_cost = BROKERAGE_FLAT + (2 * price * SLIPPAGE_PCT * LOT_SIZE)
            if expected_profit <= estimated_cost * 2:
                continue

            if (ml >= threshold and iv >= IV_MIN
                    and IV_RANK_MIN <= iv_rank <= IV_RANK_MAX):
                in_trade = True
                ep = price
                eb = bar
                eiv = iv
                esp = float(arr_spot[idx])
                edt = cur_dt
                eml = ml
                edate = date

                current_lots = get_lot_size(ml, ML_LOT_MAPPING, BASE_LOTS)

                risk_amount = ep * STOP_PCT * LOT_SIZE * current_lots
                max_risk = INITIAL_CAPITAL * MAX_CAPITAL_PER_TRADE
                if risk_amount > max_risk:
                    current_lots = max(1, int(max_risk / (ep * STOP_PCT * LOT_SIZE)))

        # ── EXIT ──
        elif in_trade:
            held = bar - eb
            pnl_pct = (price - ep) / ep
            stop = EXPIRY_STOP if expiry else STOP_PCT

            reason = None
            if pnl_pct > 0.05:
                peak_price = max(ep * (1 + pnl_pct), ep * 1.05)
                trail_stop_price = peak_price * (1 - 0.03)
                if price <= trail_stop_price:
                    reason = 'TRAIL_STOP'

            if not reason:
                if pnl_pct >= TARGET_PCT:
                    reason = 'TARGET'
                elif pnl_pct <= -stop:
                    reason = 'STOP'
                elif held >= MAX_HOLD_BARS:
                    reason = 'TIME'
                elif bar >= 67:
                    reason = 'DAY_END'

            if reason:
                gross = (price - ep) * LOT_SIZE * current_lots
                slip  = (ep + price) * SLIPPAGE_PCT * LOT_SIZE * current_lots
                net   = gross - slip - BROKERAGE_FLAT

                lot_usage[current_lots] = lot_usage.get(current_lots, 0) + 1

                if month_key not in monthly_performance:
                    monthly_performance[month_key] = 0
                monthly_performance[month_key] += net

                capital += net
                equity_curve.append((cur_dt, capital))

                trades.append({
                    'trade_num':      len(trades) + 1,
                    'date':           edate,
                    'month':          month_key,
                    'entry_datetime': edt,
                    'exit_datetime':  cur_dt,
                    'entry_price':    round(ep, 2),
                    'exit_price':     round(price, 2),
                    'lots':           current_lots,
                    'entry_iv':       round(eiv, 2),
                    'exit_iv':        round(iv, 2),
                    'iv_change':      round(iv - eiv, 2),
                    'entry_spot':     round(esp, 2),
                    'exit_spot':      round(float(arr_spot[idx]), 2),
                    'ml_signal':      round(eml, 4),
                    'bars_held':      held,
                    'pnl_pct':        round(pnl_pct * 100, 2),
                    'gross_pnl':      round(gross, 2),
                    'slip_cost':      round(slip, 2),
                    'net_pnl':        round(net, 2),
                    'capital_after':  round(capital, 2),
                    'exit_reason':    reason,
                    'is_expiry':      int(expiry),
                })

                if net < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                daily_trades[edate] = daily_trades.get(edate, 0) + 1
                daily_last_exit[edate] = bar
                in_trade = False
                current_lots = BASE_LOTS

    tdf = pd.DataFrame(trades)
    return tdf, equity_curve, lot_usage
