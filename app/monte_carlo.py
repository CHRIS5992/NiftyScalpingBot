"""
monte_carlo.py — Bootstrap resampling simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_monte_carlo(
    trade_returns: pd.Series,
    initial_capital: float,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap resample trade returns to simulate equity paths.

    Args:
        trade_returns: Series of net P&L per trade.
        initial_capital: Starting capital.
        n_simulations: Number of bootstrap simulations.
        seed: Random seed.

    Returns:
        Dict with simulation results.
    """
    np.random.seed(seed)
    n_trades = len(trade_returns)
    returns = trade_returns.values

    # Generate all simulations at once (vectorized)
    indices = np.random.randint(0, n_trades, size=(n_simulations, n_trades))
    sampled_returns = returns[indices]  # shape: (n_sims, n_trades)
    cumulative = np.cumsum(sampled_returns, axis=1)
    equity_paths = initial_capital + cumulative

    # Add starting column
    start_col = np.full((n_simulations, 1), initial_capital)
    equity_paths = np.hstack([start_col, equity_paths])

    final_values = equity_paths[:, -1]

    return {
        'equity_paths':     equity_paths,
        'final_values':     final_values,
        'median_final':     float(np.median(final_values)),
        'p5_final':         float(np.percentile(final_values, 5)),
        'p25_final':        float(np.percentile(final_values, 25)),
        'p75_final':        float(np.percentile(final_values, 75)),
        'p95_final':        float(np.percentile(final_values, 95)),
        'mean_final':       float(np.mean(final_values)),
        'std_final':        float(np.std(final_values)),
        'prob_profit':      float((final_values > initial_capital).mean() * 100),
        'prob_double':      float((final_values > 2 * initial_capital).mean() * 100),
        'worst_case':       float(np.min(final_values)),
        'best_case':        float(np.max(final_values)),
        'n_simulations':    n_simulations,
        'n_trades':         n_trades,
    }
