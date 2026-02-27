"""Quick pipeline verification with caching."""
import sys, time
sys.path.insert(0, r'd:\strem2')

from app.config import get_config_dict
from app.data_loader import load_and_clean
from app.strategy import add_features, load_cached_ml, run_backtest
from app.metrics import compute_metrics
from pathlib import Path

t0 = time.time()
cfg = get_config_dict()

print('1. Loading data...')
df_raw = load_and_clean(r'd:\strem2\FINAL_NIFTY_MASTER_ATM_5min.csv')
t1 = time.time()
print(f'   {len(df_raw):,} rows in {t1-t0:.1f}s')

print('2. Feature engineering...')
df_feat = add_features(df_raw, cfg)
df_feat = df_feat.dropna(subset=['trade_label']).reset_index(drop=True)
t2 = time.time()
print(f'   {len(df_feat):,} rows in {t2-t1:.1f}s')

print('3. Loading ML from cache...')
df_c, ml_stats = load_cached_ml(df_feat, r'd:\strem2\cache\ml_cache.pkl')
t3 = time.time()
print(f'   AUC={ml_stats["ens_auc"]:.4f} in {t3-t2:.1f}s')

print('4. Running optimized backtest...')
tdf, eq, lot_usage = run_backtest(df_c, cfg)
t4 = time.time()
print(f'   {len(tdf)} trades in {t4-t3:.1f}s')

m = compute_metrics(tdf, cfg)
t5 = time.time()

print(f'\n{"="*50}')
print(f'  Trades: {m["total_trades"]}  |  Win: {m["win_rate"]}%')
print(f'  Net P&L: Rs {m["net_pnl"]:,.2f}  |  Sharpe: {m["sharpe"]}')
print(f'  Max DD: Rs {m["max_drawdown"]:,.2f}  |  Lots: {lot_usage}')
print(f'  TOTAL TIME: {t5-t0:.1f}s')
print(f'{"="*50}')
