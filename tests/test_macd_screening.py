from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from strategy.macd_convergence import compute_macd_signals


def _cfg():
    return SimpleNamespace(
        signal=SimpleNamespace(
            prior_pos_bars=7,
            min_neg_bars=2,
            max_neg_bars=7,
            convergence_count=1,
            min_strength_ratio=2.0,
        ),
        features=SimpleNamespace(trend_score_min=99),
    )


def _signals(hist: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame({
        "trade_date": pd.bdate_range("2024-01-01", periods=len(hist)),
        "symbol": "TEST",
        "hist": hist,
        # Deliberately below the old trend gate: the exact five screening
        # rules must not silently add a sixth condition.
        "trend_score": 0,
    })
    return compute_macd_signals(frame, _cfg())


def test_exact_screen_accepts_first_convergence_after_valid_segments():
    result = _signals([1, 2, 3, 4, 3, 2, 1, -2, -1])

    row = result.iloc[-1]
    assert row["prior_pos_run"] == 7
    assert row["neg_run_k"] == 2
    assert row["convergence_count"] == 1
    assert row["prior_pos_max"] == 4
    assert row["current_neg_min"] == -2
    assert row["macd_strength_ratio"] == pytest.approx(2.0)
    assert bool(row["entry_candidate"])


@pytest.mark.parametrize(
    ("hist", "reason"),
    [
        ([1, 2, 3, 4, 3, 2, -2, -1], "prior positive run is only 6"),
        ([1, 2, 3, 4, 3, 2, 1, -1], "negative run is only 1"),
        ([1, 2, 3, 4, 3, 2, 1, -8, -7, -6, -5, -4, -3, -2, -1],
         "negative run is 8"),
        ([1, 2, 3, 4, 3, 2, 1, -3, -2], "strength ratio is below 2"),
        ([1, 2, 3, 4, 3, 2, 1, -2, -1, -0.5],
         "convergence count is 2 rather than exactly 1"),
    ],
)
def test_exact_screen_rejects_each_failed_rule(hist, reason):
    result = _signals(hist)
    assert not bool(result.iloc[-1]["entry_candidate"]), reason


def test_signal_statistics_do_not_use_future_negative_bars():
    base = [1, 2, 3, 4, 3, 2, 1, -2, -1]
    longer = base + [-100]

    before = _signals(base).iloc[-1]
    same_row_with_future = _signals(longer).iloc[-2]

    columns = [
        "prior_pos_run", "neg_run_k", "convergence_count",
        "prior_pos_max", "current_neg_min", "macd_strength_ratio",
        "entry_candidate",
    ]
    for column in columns:
        left, right = before[column], same_row_with_future[column]
        if isinstance(left, (float, np.floating)):
            assert left == pytest.approx(right)
        else:
            assert left == right

