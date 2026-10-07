"""Focused regression tests for SPA return direction and p-value selection."""
import numpy as np
import pandas as pd
import pytest

from backtest.validation import spa_test


@pytest.fixture
def monthly_returns():
    rng = np.random.default_rng(2026)
    months = pd.date_range("2010-01-31", periods=120, freq=pd.offsets.MonthEnd())
    benchmark = pd.Series(rng.normal(0.005, 0.02, len(months)), index=months)
    excess = pd.Series(rng.normal(0.04, 0.005, len(months)), index=months)
    return benchmark, excess


@pytest.mark.parametrize("direction", [1, -1], ids=["superior", "inferior"])
def test_spa_monthly_return_direction(monthly_returns, monkeypatch, direction):
    import arch.bootstrap

    real_spa = arch.bootstrap.SPA

    def seeded_spa(*args, **kwargs):
        # Seed the real bootstrap in tests only; production settings stay unchanged.
        return real_spa(*args, **kwargs, seed=42)

    monkeypatch.setattr(arch.bootstrap, "SPA", seeded_spa)
    benchmark, excess = monthly_returns
    result = spa_test(benchmark + direction * excess, benchmark, reps=199)

    assert result["p"] is not None, result["note"]
    if direction == 1:
        assert result["p"] < 0.05
    else:
        # SPA is one-sided: an inferior strategy must not reject at 5%.
        # Re-centering means its p-value need not approach 1.
        assert result["p"] > 0.05


def test_spa_selects_consistent_pvalue_by_label(monthly_returns, monkeypatch):
    import arch.bootstrap

    # Real single-model SPA p-values can coincide. Replace only the output
    # property to expose a positional selection bug, retaining real computation.
    monkeypatch.setattr(
        arch.bootstrap.SPA,
        "pvalues",
        property(lambda self: pd.Series(
            [0.01, 0.99, 0.37], index=["lower", "upper", "consistent"]
        )),
    )
    benchmark, excess = monthly_returns
    result = spa_test(benchmark + excess, benchmark, reps=199)

    assert result["p"] == 0.37, result["note"]
