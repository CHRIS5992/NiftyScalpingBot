"""
data_loader.py — Data loading and cleaning.
Exact reproduction of notebook Cell 2 load_and_clean().
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


def load_and_clean(filepath: str | Path) -> pd.DataFrame:
    """Load CSV and clean exactly as notebook Cell 2.
    
    Steps (verbatim):
      1. Read CSV
      2. Parse 'datetime' column
      3. Sort by datetime, reset index
      4. Compute Straddle_Price if missing
      5. Filter iv ∈ (0, 100)
      6. Filter Straddle_Price > 5
      7. Filter session 09:20–15:00
      8. Add 'date' as string column
      9. Reset index
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {filepath}')

    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    if 'Straddle_Price' not in df.columns:
        df['Straddle_Price'] = df['CE_close'] + df['PE_close']

    df = df[(df['iv'] > 0) & (df['iv'] < 100)].copy()
    df = df[df['Straddle_Price'] > 5].copy()

    t = df['datetime'].dt.time
    df = df[(t >= dt.time(9, 20)) & (t <= dt.time(15, 0))].copy()
    df['date'] = df['datetime'].dt.date.astype(str)
    df = df.reset_index(drop=True)

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
    }
