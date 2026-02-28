"""
config.py — All strategy constants and parameters.
Matched to Final_straddle.ipynb (Regime-Switch ML Bot).
"""

# ----------------------------------------------------------------
# CAPITAL & POSITION SIZING
# ----------------------------------------------------------------
INITIAL_CAPITAL: float = 1_000_000  # Rs 10,00,000
LOT_SIZE: int = 65                  # NIFTY lot size

# Risk management
MAX_RISK_PER_TRADE: float = 0.01    # 1% risk per trade

# ----------------------------------------------------------------
# STRATEGY PARAMETERS (Regime-Switch Bot)
# ----------------------------------------------------------------
STOP_LOSS_PCT: float = -0.0015           # -0.15% Nifty Spot move
TRAILING_STOP_ACTIVATION: float = 0.0012 # Activate trail at +0.12%
TRAILING_STOP_DISTANCE: float = 0.0005   # Trail by 0.05%
MAX_HOLD_BARS: int = 12                  # 60 minutes max
MIN_EXPECTED_RETURN: float = 0.0001      # Min expected return
MIN_SIGNAL_PROB: float = 0.32            # ML confidence threshold

# ----------------------------------------------------------------
# ML SETTINGS (Walk-Forward)
# ----------------------------------------------------------------
N_SPLITS: int = 5
N_ESTIMATORS: int = 100
MAX_DEPTH: int = 5
RANDOM_STATE: int = 42
ML_EVAL_SIGNAL_PROB: float = 0.50  # Notebook Cell 3: threshold for fold accuracy eval

# ----------------------------------------------------------------
# FEATURE ENGINEERING (5-MIN BARS)
# ----------------------------------------------------------------
HV_SHORT_WINDOW: int = 75
HV_LONG_WINDOW: int = 225
MOMENTUM_THRESHOLD: float = 0.0015

# ----------------------------------------------------------------
# COST ASSUMPTIONS
# ----------------------------------------------------------------
SLIPPAGE_RATE: float = 0.001      # 0.1% of trade value
BROKERAGE_PER_TRADE: float = 40   # Rs 40 flat per round-trip

# ----------------------------------------------------------------
# REGIME NAMES & COLORS
# ----------------------------------------------------------------
REGIME_MAP: dict = {0: 'Trend HV', 1: 'Trend LV', 2: 'Range HV', 3: 'Range LV'}
REGIME_COLORS: dict = {0: '#d62728', 1: '#2ca02c', 2: '#ff7f0e', 3: '#1f77b4'}

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
        'MAX_RISK_PER_TRADE': MAX_RISK_PER_TRADE,
        'STOP_LOSS_PCT': STOP_LOSS_PCT,
        'TRAILING_STOP_ACTIVATION': TRAILING_STOP_ACTIVATION,
        'TRAILING_STOP_DISTANCE': TRAILING_STOP_DISTANCE,
        'MAX_HOLD_BARS': MAX_HOLD_BARS,
        'MIN_EXPECTED_RETURN': MIN_EXPECTED_RETURN,
        'MIN_SIGNAL_PROB': MIN_SIGNAL_PROB,
        'N_SPLITS': N_SPLITS,
        'N_ESTIMATORS': N_ESTIMATORS,
        'MAX_DEPTH': MAX_DEPTH,
        'RANDOM_STATE': RANDOM_STATE,
        'ML_EVAL_SIGNAL_PROB': ML_EVAL_SIGNAL_PROB,
        'HV_SHORT_WINDOW': HV_SHORT_WINDOW,
        'HV_LONG_WINDOW': HV_LONG_WINDOW,
        'MOMENTUM_THRESHOLD': MOMENTUM_THRESHOLD,
        'SLIPPAGE_RATE': SLIPPAGE_RATE,
        'BROKERAGE_PER_TRADE': BROKERAGE_PER_TRADE,
        'REGIME_MAP': REGIME_MAP,
        'REGIME_COLORS': REGIME_COLORS,
    }
    cfg.update(overrides)
    return cfg
