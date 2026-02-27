"""
config.py — All strategy constants and parameters.
Extracted verbatim from NIFTY Straddle Bot v3.0 notebook Cell 1.
"""

# ----------------------------------------------------------------
# CAPITAL & POSITION SIZING
# ----------------------------------------------------------------
INITIAL_CAPITAL: float = 1_000_000  # Rs 10,00,000
LOT_SIZE: int = 65                  # NIFTY lot size (2026 NSE)
BASE_LOTS: int = 1                  # Base lots per trade

# Dynamic lot sizing based on ML confidence
ML_LOT_MAPPING: list[dict] = [
    {'min_ml': 0.45, 'max_ml': 0.55, 'lots': 1, 'color': '#FFA500'},
    {'min_ml': 0.55, 'max_ml': 0.70, 'lots': 2, 'color': '#00E676'},
    {'min_ml': 0.70, 'max_ml': 1.00, 'lots': 3, 'color': '#40C4FF'},
]

# Risk management
MAX_LOTS_PER_TRADE: int = 3
MAX_CAPITAL_PER_TRADE: float = 0.02  # Max 2% of capital at risk

# ----------------------------------------------------------------
# STRATEGY PARAMETERS (v4.0 - Optimized)
# ----------------------------------------------------------------
ML_THRESHOLD: float = 0.46
IV_ZSCORE_BOOST: float = 0.5
IV_MIN: float = 8.5
IV_RANK_MIN: float = 0.15
IV_RANK_MAX: float = 0.85
TARGET_PCT: float = 0.14       # +14% straddle -> book profit
STOP_PCT: float = 0.07         # -7%  straddle -> cut loss
EXPIRY_STOP: float = 0.05      # -5% tighter stop on Tuesdays (expiry)
MAX_HOLD_BARS: int = 9         # 9 x 5min = 45 min
MAX_TRADES_PER_DAY: int = 3
COOLDOWN_BARS: int = 4
SESSION_START_BAR: int = 3     # skip first 3 bars (open noise)
SESSION_END_BAR: int = 62      # no new entries after bar 62
LOSS_STREAK_LIMIT: int = 3     # Stop after 3 consecutive losses

# ----------------------------------------------------------------
# ML SETTINGS
# ----------------------------------------------------------------
TRAIN_RATIO: float = 0.80
RF_TREES: int = 300
GB_TREES: int = 200
RF_MAX_DEPTH: int = 8
RF_MIN_SAMPLES_LEAF: int = 15
GB_MAX_DEPTH: int = 4
GB_LEARNING_RATE: float = 0.05
GB_SUBSAMPLE: float = 0.8
RANDOM_STATE: int = 42
RF_WEIGHT: float = 0.50
GB_WEIGHT: float = 0.50

# ----------------------------------------------------------------
# COST ASSUMPTIONS
# ----------------------------------------------------------------
SLIPPAGE_PCT: float = 0.001    # 0.1% of premium per leg
BROKERAGE_FLAT: float = 40     # Rs 40 flat per round-trip
TAX_RATE: float = 0.15         # 15% approximate taxes on gross

# ----------------------------------------------------------------
# TIMEFRAME PARAMS (5min)
# ----------------------------------------------------------------
USE_TIMEFRAME: str = '5min'
FUTURE_BARS: int = 3
RS: int = 3      # short rolling window
RM: int = 6      # medium rolling window
RL: int = 12     # long rolling window
ROLL_RANK: int = 78

# ----------------------------------------------------------------
# FEATURE LIST (45 features)
# ----------------------------------------------------------------
FEATURES: list[str] = [
    'iv_lag1', 'iv_lag2', 'iv_lag3', 'iv_lag5',
    'iv_ma_s', 'iv_ma_m', 'iv_ma_l', 'iv_std_s', 'iv_std_m', 'iv_std_l',
    'iv_z', 'iv_rank', 'iv_pct1', 'iv_pct_s', 'iv_diff1', 'iv_diff_s',
    'iv_ma_cross', 'iv_accel', 'iv_z_burst',
    'prem_ret1', 'prem_ret_s', 'prem_ma_ratio', 'prem_std_s',
    'spot_ret1', 'spot_ret_s', 'spot_ret_m', 'spot_rv_s', 'spot_rv_m',
    'rv_iv_ratio', 'gamma_proxy',
    'total_oi', 'oi_ratio', 'oi_chg1', 'oi_chg_s',
    'total_vol', 'vol_surge', 'cp_vol_ratio', 'pcp',
    'hour', 'minute', 'dow', 'is_expiry', 'time_from_open', 'bar_num',
    'intraday_range',
]

# ----------------------------------------------------------------
# COLOR PALETTE
# ----------------------------------------------------------------
GOLD   = '#FFD700'
GREEN  = '#00E676'
RED    = '#FF5252'
DARK   = '#0d1117'
BLUE   = '#40C4FF'
ORANGE = '#FF9800'
PURPLE = '#CE93D8'


def get_config_dict(**overrides) -> dict:
    """Return all config as a dict, applying any overrides from sidebar."""
    cfg = {
        'INITIAL_CAPITAL': INITIAL_CAPITAL,
        'LOT_SIZE': LOT_SIZE,
        'BASE_LOTS': BASE_LOTS,
        'ML_LOT_MAPPING': ML_LOT_MAPPING,
        'MAX_LOTS_PER_TRADE': MAX_LOTS_PER_TRADE,
        'MAX_CAPITAL_PER_TRADE': MAX_CAPITAL_PER_TRADE,
        'ML_THRESHOLD': ML_THRESHOLD,
        'IV_ZSCORE_BOOST': IV_ZSCORE_BOOST,
        'IV_MIN': IV_MIN,
        'IV_RANK_MIN': IV_RANK_MIN,
        'IV_RANK_MAX': IV_RANK_MAX,
        'TARGET_PCT': TARGET_PCT,
        'STOP_PCT': STOP_PCT,
        'EXPIRY_STOP': EXPIRY_STOP,
        'MAX_HOLD_BARS': MAX_HOLD_BARS,
        'MAX_TRADES_PER_DAY': MAX_TRADES_PER_DAY,
        'COOLDOWN_BARS': COOLDOWN_BARS,
        'SESSION_START_BAR': SESSION_START_BAR,
        'SESSION_END_BAR': SESSION_END_BAR,
        'LOSS_STREAK_LIMIT': LOSS_STREAK_LIMIT,
        'TRAIN_RATIO': TRAIN_RATIO,
        'RF_TREES': RF_TREES,
        'GB_TREES': GB_TREES,
        'RF_MAX_DEPTH': RF_MAX_DEPTH,
        'RF_MIN_SAMPLES_LEAF': RF_MIN_SAMPLES_LEAF,
        'GB_MAX_DEPTH': GB_MAX_DEPTH,
        'GB_LEARNING_RATE': GB_LEARNING_RATE,
        'GB_SUBSAMPLE': GB_SUBSAMPLE,
        'RANDOM_STATE': RANDOM_STATE,
        'RF_WEIGHT': RF_WEIGHT,
        'GB_WEIGHT': GB_WEIGHT,
        'SLIPPAGE_PCT': SLIPPAGE_PCT,
        'BROKERAGE_FLAT': BROKERAGE_FLAT,
        'TAX_RATE': TAX_RATE,
        'USE_TIMEFRAME': USE_TIMEFRAME,
        'FUTURE_BARS': FUTURE_BARS,
        'RS': RS, 'RM': RM, 'RL': RL,
        'ROLL_RANK': ROLL_RANK,
        'FEATURES': FEATURES,
    }
    cfg.update(overrides)
    return cfg
