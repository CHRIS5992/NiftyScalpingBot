"""
pretrain.py — Run ML training ONCE, cache results to disk.
After running this, the Streamlit app loads in ~5 seconds instead of 5 minutes.

Usage:
    python pretrain.py
"""
import sys, time, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_config_dict
from app.data_loader import load_and_clean
from app.strategy import add_features, train_ml_models

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

t0 = time.time()
cfg = get_config_dict()

print('1. Loading data...')
df_raw = load_and_clean(Path(__file__).parent / 'FINAL_NIFTY_MASTER_ATM_5min.csv')
t1 = time.time()
print(f'   {len(df_raw):,} rows in {t1-t0:.1f}s')

print('2. Feature engineering...')
df_feat = add_features(df_raw, cfg)
df_feat = df_feat.dropna(subset=['trade_label']).reset_index(drop=True)
t2 = time.time()
print(f'   {len(df_feat):,} rows in {t2-t1:.1f}s')

print('3. Training ML (RF 300 + GBM 200) — this takes a few minutes...')
df_c, rf, gb, scaler, col_means, ml_stats = train_ml_models(df_feat, cfg)
t3 = time.time()
print(f'   Ensemble AUC = {ml_stats["ens_auc"]:.4f} in {t3-t2:.1f}s')

# Save everything needed to skip training
print('4. Saving cache...')
cache = {
    'ml_signal': df_c['ml_signal'].values,
    'ml_stats': {k: v for k, v in ml_stats.items() if k != 'feat_imp'},
    'feat_imp_index': ml_stats['feat_imp'].index.tolist(),
    'feat_imp_values': ml_stats['feat_imp'].values.tolist(),
}
with open(CACHE_DIR / 'ml_cache.pkl', 'wb') as f:
    pickle.dump(cache, f)

t4 = time.time()
print(f'\n{"="*60}')
print(f'  Cache saved to: {CACHE_DIR / "ml_cache.pkl"}')
print(f'  ML signal shape: {cache["ml_signal"].shape}')
print(f'  Total time: {t4-t0:.1f}s')
print(f'  Next app load will skip ML training entirely!')
print(f'{"="*60}')
