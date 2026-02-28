"""
strategy.py — Feature engineering, ML model training (walk-forward), and backtest engine.
Exact reproduction of Final_straddle.ipynb Cells 4, 5.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

# Use local cache/ dir if writable, else fall back to /tmp (Streamlit Cloud)
_LOCAL_CACHE = Path(__file__).resolve().parent.parent / 'cache'
try:
    _LOCAL_CACHE.mkdir(exist_ok=True)
    CACHE_DIR = _LOCAL_CACHE
except OSError:
    import tempfile
    CACHE_DIR = Path(tempfile.gettempdir()) / 'straddle_cache'
    CACHE_DIR.mkdir(exist_ok=True)


# ============================================================
# FEATURE ENGINEERING — Exact Cell 4
# ============================================================

def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Engineer all features exactly as notebook Cell 4.

    Returns DataFrame with all features, market_regime, and target.
    """
    df = df.copy()

    ANNUALIZE_FACTOR = np.sqrt(18900)
    HV_SHORT = cfg.get('HV_SHORT_WINDOW', 75)
    HV_LONG = cfg.get('HV_LONG_WINDOW', 225)
    MOMENTUM_THRESHOLD = cfg.get('MOMENTUM_THRESHOLD', 0.0015)

    # 1. VOLATILITY REGIME
    df['hv_short'] = df['returns'].rolling(HV_SHORT).std() * ANNUALIZE_FACTOR
    df['hv_long'] = df['returns'].rolling(HV_LONG).std() * ANNUALIZE_FACTOR
    df['vol_regime'] = (df['hv_short'] > df['hv_long'] * 1.1).astype(int)

    # 2. TREND STRENGTH
    df['up_move'] = df['returns'].clip(lower=0)
    df['down_move'] = (-df['returns']).clip(lower=0)
    df['up_strength'] = df['up_move'].rolling(20).mean()
    df['down_strength'] = df['down_move'].rolling(20).mean()
    df['trend_strength'] = abs(df['up_strength'] - df['down_strength']) / (
        df['up_strength'] + df['down_strength'] + 1e-8
    )
    df['trend_direction'] = (df['up_strength'] > df['down_strength']).astype(int)

    # 3. MOMENTUM
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(period)
        df[f'momentum_rank_{period}'] = df[f'momentum_{period}'].rolling(60).rank(pct=True)

    # 4. MEAN REVERSION
    df['z_score_10'] = (df['close'] - df['close'].rolling(10).mean()) / (
        df['close'].rolling(10).std() + 1e-8
    )
    df['z_score_20'] = (df['close'] - df['close'].rolling(20).mean()) / (
        df['close'].rolling(20).std() + 1e-8
    )

    # 5. VOLUME
    if 'CE_volume' in df.columns and 'PE_volume' in df.columns:
        df['total_volume'] = df['CE_volume'] + df['PE_volume']
        df['volume_ratio'] = df['total_volume'] / (df['total_volume'].rolling(20).mean() + 1e-8)
    else:
        df['volume_ratio'] = 1.0

    # 6. MARKET REGIME
    conditions = [
        (df['trend_direction'] == 1) & (df['vol_regime'] == 1),
        (df['trend_direction'] == 1) & (df['vol_regime'] == 0),
        (df['trend_direction'] == 0) & (df['vol_regime'] == 1),
        (df['trend_direction'] == 0) & (df['vol_regime'] == 0),
    ]
    df['market_regime'] = np.select(conditions, [0, 1, 2, 3], default=0)

    # 7. TARGET
    df['forward_short'] = df['close'].shift(-6) / df['close'] - 1
    df['forward_long'] = df['close'].shift(-12) / df['close'] - 1

    df['target'] = 0
    mask_trend = (df['market_regime'].isin([0, 1])) & (df['forward_short'] > MOMENTUM_THRESHOLD)
    mask_ranging = (df['market_regime'].isin([2, 3])) & (
        df['forward_long'] > MOMENTUM_THRESHOLD * 0.8
    )
    df.loc[mask_trend | mask_ranging, 'target'] = 1

    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_columns(df_feat: pd.DataFrame) -> list[str]:
    """Get the list of feature columns (exclude metadata and targets).

    Matches notebook Cell 3 exactly — the notebook's data loader only creates
    'close', 'returns', 'date', 'dayofweek'. It does NOT create 'high', 'low',
    'hour', 'minute'. Since our data_loader creates those for display use,
    we exclude them here to match the notebook's 32 features.
    """
    exclude_cols = [
        'datetime', 'date', 'time', 'symbol', 'ticker',
        'close', 'returns', 'target', 'forward_short',
        'forward_long', 'market_regime', 'up_move',
        'down_move', 'total_volume',
        # Not created by notebook Cell 3 loader — exclude to match 32 features
        'high', 'low', 'hour', 'minute',
    ]
    feature_cols = [c for c in df_feat.columns if c not in exclude_cols]
    # Only keep numeric columns (notebook: select_dtypes(include=[np.number]))
    numeric_cols = df_feat[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols


# ============================================================
# ML CACHING HELPERS
# ============================================================

def _make_cache_key(df_feat: pd.DataFrame, cfg: dict, feature_cols: list[str]) -> str:
    """Deterministic hash of data shape + ML config for cache lookup."""
    sig = hashlib.md5()
    sig.update(str(len(df_feat)).encode())
    sig.update(str(sorted(feature_cols)).encode())
    sig.update(str(df_feat['target'].sum()).encode())
    for k in ['N_SPLITS', 'N_ESTIMATORS', 'MAX_DEPTH', 'RANDOM_STATE', 'ML_EVAL_SIGNAL_PROB']:
        sig.update(f'{k}={cfg.get(k)}'.encode())
    return sig.hexdigest()[:16]


def _load_cached(cache_key: str):
    """Load cached ML artifacts if they exist."""
    if not _HAS_JOBLIB:
        return None
    cache_path = CACHE_DIR / f'ml_{cache_key}.joblib'
    if cache_path.exists():
        return joblib.load(cache_path)
    return None


def _save_cache(cache_key: str, payload: dict):
    """Persist ML artifacts to disk."""
    if not _HAS_JOBLIB:
        return
    cache_path = CACHE_DIR / f'ml_{cache_key}.joblib'
    joblib.dump(payload, cache_path, compress=3)


# ============================================================
# ML TRAINING — Walk-Forward (Exact Cell 3)
# ============================================================

def train_ml_models(df_feat: pd.DataFrame, cfg: dict) -> tuple:
    """Train RandomForest with TimeSeriesSplit walk-forward.

    Uses joblib disk cache — subsequent runs with the same data & config
    return instantly instead of retraining.

    Returns: (df_feat_with_signal, models, scalers, feature_cols, fold_results)
    """
    N_SPLITS = cfg.get('N_SPLITS', 5)
    N_ESTIMATORS = cfg.get('N_ESTIMATORS', 100)
    MAX_DEPTH = cfg.get('MAX_DEPTH', 5)
    RANDOM_STATE = cfg.get('RANDOM_STATE', 42)
    # Notebook Cell 3 uses 0.50 for fold accuracy evaluation (separate from backtest's 0.32)
    MIN_SIGNAL_PROB = cfg.get('ML_EVAL_SIGNAL_PROB', 0.50)

    feature_cols = get_feature_columns(df_feat)
    X = df_feat[feature_cols].copy()
    y = df_feat['target'].copy()

    # ── Check disk cache ──
    cache_key = _make_cache_key(df_feat, cfg, feature_cols)
    cached = _load_cached(cache_key)
    if cached is not None:
        models = cached['models']
        scalers = cached['scalers']
        fold_results = cached['fold_results']
        final_model = models[-1]
        final_scaler = scalers[-1]
        X_full_scaled = final_scaler.transform(X)
        df_feat = df_feat.copy()
        df_feat['ml_signal'] = final_model.predict_proba(X_full_scaled)[:, 1]
        return df_feat, models, scalers, feature_cols, fold_results

    # ── Train from scratch ──
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    models, scalers, fold_results = [], [], []
    fold_number = 1

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = (probs > MIN_SIGNAL_PROB).astype(int)
        accuracy = (preds == y_test.values).mean()
        signal_rate = preds.mean()

        fold_results.append({
            'fold': fold_number,
            'accuracy': accuracy,
            'signal_rate': signal_rate,
        })
        models.append(model)
        scalers.append(scaler)
        fold_number += 1

    # Generate ML signals using final fold model
    final_model = models[-1]
    final_scaler = scalers[-1]
    X_full_scaled = final_scaler.transform(X)
    df_feat = df_feat.copy()
    df_feat['ml_signal'] = final_model.predict_proba(X_full_scaled)[:, 1]

    # ── Save to disk cache ──
    _save_cache(cache_key, {
        'models': models,
        'scalers': scalers,
        'fold_results': fold_results,
    })

    return df_feat, models, scalers, feature_cols, fold_results


# ============================================================
# RISK MANAGER — Exact Cell 5
# ============================================================

class RiskManager:
    """Dynamic risk management matching notebook exactly."""

    def __init__(self, initial_capital: float, cfg: dict):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_drawdown = 0
        self.consecutive_losses = 0
        self.trades = []
        self.cfg = cfg

    def calculate_position_size(self, entry_price: float, signal_strength: float, regime: int) -> int:
        LOT_SIZE = self.cfg['LOT_SIZE']
        MAX_RISK = self.cfg['MAX_RISK_PER_TRADE']
        STOP_LOSS = self.cfg['STOP_LOSS_PCT']

        base_risk = self.current_capital * MAX_RISK
        strength_multiplier = 0.5 + signal_strength

        regime_multiplier = {
            0: 1.0,
            1: 1.2,
            2: 0.7,
            3: 0.0,  # Block Range LV
        }.get(regime, 0.0)

        performance_multiplier = max(0.6, 1.0 - (self.consecutive_losses * 0.10))
        risk_amount = base_risk * strength_multiplier * regime_multiplier * performance_multiplier

        stop_distance_points = entry_price * abs(STOP_LOSS)
        lots = int(risk_amount / (stop_distance_points * LOT_SIZE))

        return max(1, min(lots, 50))

    def add_trade(self, trade: dict):
        self.trades.append(trade)
        pnl = trade['net_pnl']
        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        current_dd = (self.peak_capital - self.current_capital) / self.peak_capital
        self.max_drawdown = max(self.max_drawdown, current_dd)
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def should_stop_trading(self) -> bool:
        current_dd = (self.peak_capital - self.current_capital) / self.peak_capital
        return current_dd > 0.25

    def get_summary(self) -> dict:
        if not self.trades:
            return {}
        df_trades = pd.DataFrame(self.trades)
        return {
            'total_trades': len(df_trades),
            'win_rate': (df_trades['net_pnl'] > 0).mean() * 100,
            'total_pnl': df_trades['net_pnl'].sum(),
            'avg_win': df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean()
                       if any(df_trades['net_pnl'] > 0) else 0,
            'avg_loss': df_trades[df_trades['net_pnl'] < 0]['net_pnl'].mean()
                        if any(df_trades['net_pnl'] < 0) else 0,
            'max_drawdown': self.max_drawdown * 100,
            'final_capital': self.current_capital,
            'roi': (self.current_capital - self.initial_capital) / self.initial_capital * 100,
        }


# ============================================================
# BACKTEST ENGINE — Exact Cell 5
# ============================================================

def run_backtest(df: pd.DataFrame, cfg: dict) -> tuple:
    """Run the full backtest exactly as notebook Cell 5.

    Returns: (trades_df, summary_dict, risk_manager)
    """
    INITIAL_CAPITAL = cfg['INITIAL_CAPITAL']
    LOT_SIZE = cfg['LOT_SIZE']
    STOP_LOSS_PCT = cfg['STOP_LOSS_PCT']
    TRAILING_STOP_ACTIVATION = cfg['TRAILING_STOP_ACTIVATION']
    TRAILING_STOP_DISTANCE = cfg['TRAILING_STOP_DISTANCE']
    MAX_HOLD_BARS = cfg['MAX_HOLD_BARS']
    MIN_EXPECTED_RETURN = cfg['MIN_EXPECTED_RETURN']
    MIN_SIGNAL_PROB = cfg['MIN_SIGNAL_PROB']

    risk_mgr = RiskManager(INITIAL_CAPITAL, cfg)
    in_position = False
    entry_price = entry_idx = entry_regime = entry_signal = 0
    position_lots = 0
    trailing_stop = None

    test_indices = df.index.tolist()

    for i in test_indices:
        row = df.loc[i]

        if risk_mgr.should_stop_trading():
            break

        # EXIT LOGIC
        if in_position:
            current_price = row['close']
            bars_held = i - entry_idx
            pnl_pct = (current_price - entry_price) / entry_price

            exit_signal = False
            exit_reason = ""

            if pnl_pct < STOP_LOSS_PCT:
                exit_signal = True
                exit_reason = "STOP_LOSS"
            elif pnl_pct > TRAILING_STOP_ACTIVATION:
                if trailing_stop is None:
                    trailing_stop = current_price * (1 - TRAILING_STOP_DISTANCE)
                elif current_price < trailing_stop:
                    exit_signal = True
                    exit_reason = "TRAILING_STOP"
                else:
                    trailing_stop = max(trailing_stop, current_price * (1 - TRAILING_STOP_DISTANCE))
            elif bars_held >= MAX_HOLD_BARS:
                exit_signal = True
                exit_reason = "TIME_EXIT"
            elif row['market_regime'] != entry_regime and pnl_pct > 0:
                exit_signal = True
                exit_reason = "REGIME_CHANGE"

            if exit_signal:
                gross_pnl = (current_price - entry_price) * LOT_SIZE * position_lots
                slippage = abs(gross_pnl) * cfg['SLIPPAGE_RATE']
                net_pnl = gross_pnl - slippage - cfg['BROKERAGE_PER_TRADE']

                trade = {
                    'trade_num': len(risk_mgr.trades) + 1,
                    'entry_date': str(df.loc[entry_idx]['date']),
                    'exit_date': str(row['date']),
                    'entry_time': df.loc[entry_idx]['datetime'],
                    'exit_time': row['datetime'],
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(current_price, 2),
                    'lots': position_lots,
                    'gross_pnl': round(gross_pnl, 2),
                    'slippage': round(slippage, 2),
                    'net_pnl': round(net_pnl, 2),
                    'pnl_pct': round(pnl_pct * 100, 4),
                    'exit_reason': exit_reason,
                    'regime': entry_regime,
                    'signal': round(entry_signal, 4),
                    'bars_held': bars_held,
                }

                risk_mgr.add_trade(trade)
                in_position = False
                trailing_stop = None

        # ENTRY LOGIC
        if not in_position:
            signal = row['ml_signal']
            regime = row['market_regime']

            if regime == 3:
                continue  # Skip Range LV

            if signal > MIN_SIGNAL_PROB:
                expected_return = signal * 0.005

                if expected_return > MIN_EXPECTED_RETURN:
                    position_lots = risk_mgr.calculate_position_size(
                        row['close'], signal, regime
                    )
                    in_position = True
                    entry_price = row['close']
                    entry_idx = i
                    entry_regime = regime
                    entry_signal = signal

    trades_df = pd.DataFrame(risk_mgr.trades)
    summary = risk_mgr.get_summary()

    return trades_df, summary, risk_mgr
