"""三类趋势标签及分情景表现汇总。"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_ORDER = ["up", "down", "sideways"]


def classify_trend(
    close: pd.Series,
    fast_window: int = 20,
    slow_window: int = 60,
    slope_window: int = 20,
) -> pd.Series:
    """只用当前及历史收盘价划分 up/down/sideways。"""
    values = pd.to_numeric(close, errors="coerce")
    fast = values.rolling(fast_window, min_periods=fast_window).mean()
    slow = values.rolling(slow_window, min_periods=slow_window).mean()
    slope = slow.pct_change(slope_window, fill_method=None)

    up = (values > slow) & (fast > slow) & (slope > 0)
    down = (values < slow) & (fast < slow) & (slope < 0)
    labels = np.select([up, down], ["up", "down"], default="sideways")
    return pd.Series(labels, index=close.index, dtype="object")


def summarize_market_regime(
    equity_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "market_regime", "n_days", "cumulative_return", "annualized_return",
        "annualized_volatility", "sharpe", "win_rate",
    ]
    if equity_df.empty or "market_regime" not in feature_df.columns:
        return pd.DataFrame(columns=columns)

    labels = (
        feature_df[["trade_date", "market_regime"]]
        .drop_duplicates("trade_date", keep="last")
    )
    daily = (
        equity_df[["trade_date", "equity"]]
        .sort_values("trade_date")
        .assign(daily_return=lambda x: x["equity"].pct_change().fillna(0.0))
        .merge(labels, on="trade_date", how="left")
    )
    daily["market_regime"] = daily["market_regime"].fillna("sideways")

    rows = []
    for label in REGIME_ORDER:
        returns = daily.loc[daily["market_regime"] == label, "daily_return"].dropna()
        n = len(returns)
        cumulative = float((1.0 + returns).prod() - 1.0) if n else np.nan
        annualized = (
            float((1.0 + cumulative) ** (252.0 / n) - 1.0)
            if n and cumulative > -1 else np.nan
        )
        volatility = float(returns.std(ddof=1) * np.sqrt(252)) if n > 1 else np.nan
        sharpe = (
            float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
            if n > 1 and returns.std(ddof=1) > 0 else np.nan
        )
        rows.append({
            "market_regime": label,
            "n_days": n,
            "cumulative_return": cumulative,
            "annualized_return": annualized,
            "annualized_volatility": volatility,
            "sharpe": sharpe,
            "win_rate": float((returns > 0).mean()) if n else np.nan,
        })
    return pd.DataFrame(rows, columns=columns)


def summarize_stock_trend(trades_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "stock_trend", "n_trades", "average_return", "win_rate", "realized_pnl",
    ]
    if trades_df is None or trades_df.empty or "entry_stock_trend" not in trades_df.columns:
        return pd.DataFrame(columns=columns)

    rows = []
    for label in REGIME_ORDER:
        group = trades_df[trades_df["entry_stock_trend"] == label]
        rows.append({
            "stock_trend": label,
            "n_trades": len(group),
            "average_return": float(group["return"].mean()) if len(group) else np.nan,
            "win_rate": float((group["return"] > 0).mean()) if len(group) else np.nan,
            "realized_pnl": float(group["realized_pnl"].sum()) if len(group) else 0.0,
        })
    return pd.DataFrame(rows, columns=columns)
