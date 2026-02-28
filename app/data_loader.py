"""
data_loader.py — Data loading and cleaning.
Exact reproduction of Final_straddle.ipynb Cell 2 & Cell 4 load_and_prepare_data().
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def load_and_clean(filepath: str | Path) -> pd.DataFrame:
    """Load CSV and prepare data exactly as notebook.

    Steps:
      1. Read CSV, parse datetime
      2. Sort by datetime, reset index
      3. Map 'spot' or 'Straddle_Price' -> 'close'
      4. Compute returns
      5. Add date features
      6. Drop NaN, reset index
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {filepath}')

    df = pd.read_csv(path)

    if 'datetime' not in df.columns:
        raise ValueError("'datetime' column not found in dataset.")

    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset=['datetime'])

    # Price selection — exactly as notebook
    if 'spot' in df.columns:
        df['close'] = df['spot']
    elif 'Straddle_Price' in df.columns:
        df['close'] = df['Straddle_Price']
    else:
        raise ValueError("Neither 'spot' nor 'Straddle_Price' found.")

    df = df[df['close'] > 0]

    # Returns
    df['returns'] = df['close'].pct_change()

    # High/Low for ATR (safe fallback)
    if 'high' not in df.columns:
        df['high'] = df['close']
    if 'low' not in df.columns:
        df['low'] = df['close']

    # Date features
    df['date'] = df['datetime'].dt.date.astype(str)
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute

    df = df.dropna().reset_index(drop=True)
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return summary statistics about the loaded data."""
    return {
        'total_rows': len(df),
        'start_date': df['date'].min(),
        'end_date': df['date'].max(),
        'trading_days': df['date'].nunique(),
        'columns': list(df.columns),
        'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
        'price_col': 'spot' if 'spot' in df.columns else 'Straddle_Price',
    }
