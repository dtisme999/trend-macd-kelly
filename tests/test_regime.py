from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.regime import (
    classify_trend,
    summarize_market_regime,
    summarize_stock_trend,
)


def test_classify_trend_emits_exactly_three_labels():
    values = (
        list(np.linspace(100, 130, 90))
        + list(np.linspace(130, 80, 90))
        + [80.0] * 90
    )
    labels = classify_trend(
        pd.Series(values), fast_window=5, slow_window=10, slope_window=3,
    )

    assert labels.iloc[40] == "up"
    assert labels.iloc[150] == "down"
    assert labels.iloc[-1] == "sideways"
    assert set(labels.unique()) == {"up", "down", "sideways"}


def test_market_regime_summary_is_recomputable_from_daily_returns():
    dates = pd.bdate_range("2024-01-01", periods=6)
    equity = pd.DataFrame({
        "trade_date": dates,
        "equity": [100, 110, 99, 108.9, 108.9, 119.79],
    })
    features = pd.DataFrame({
        "trade_date": np.repeat(dates, 2),
        "symbol": ["A", "B"] * 6,
        "market_regime": np.repeat(["up", "up", "down", "down", "sideways", "sideways"], 2),
    })

    summary = summarize_market_regime(equity, features).set_index("market_regime")

    assert set(summary.index) == {"up", "down", "sideways"}
    assert summary.loc["up", "n_days"] == 2
    assert summary.loc["down", "n_days"] == 2
    assert summary.loc["sideways", "n_days"] == 2
    # Returns assigned to labels of their ending date: up=[0,10%],
    # down=[-10%,10%], sideways=[0%,10%].
    assert summary.loc["up", "cumulative_return"] == pytest.approx(0.10)
    assert summary.loc["down", "cumulative_return"] == pytest.approx(-0.01)
    assert summary.loc["sideways", "cumulative_return"] == pytest.approx(0.10)


def test_stock_trend_summary_groups_entry_time_trade_results():
    trades = pd.DataFrame({
        "entry_stock_trend": ["up", "up", "down", "sideways"],
        "return": [0.10, -0.05, -0.20, 0.02],
        "realized_pnl": [100, -50, -200, 20],
    })

    summary = summarize_stock_trend(trades).set_index("stock_trend")

    assert summary.loc["up", "n_trades"] == 2
    assert summary.loc["up", "win_rate"] == pytest.approx(0.5)
    assert summary.loc["up", "average_return"] == pytest.approx(0.025)
    assert summary.loc["down", "realized_pnl"] == pytest.approx(-200)

