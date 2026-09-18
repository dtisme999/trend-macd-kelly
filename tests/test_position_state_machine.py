from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from backtest.executor import run_executor
from config_loader import load_config
from strategy import rules


def _pos(state=1, *, entry=10, added=20, **overrides):
    values = {
        "A": 100.0,
        "qty": 1_000,
        "state": state,
        "entry_seq_idx": entry,
        "add_seq_idx": added,
        "reached_110": False,
        "reached_120": False,
        "reached_130": False,
        "reached_140": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _action(pos, ratio, seq_idx):
    return rules.check_state_action(
        pos, close=pos.A * ratio, seq_idx=seq_idx,
        cfg_risk=load_config().risk,
    )


@pytest.mark.parametrize(
    ("state", "ratio", "seq_idx", "reason", "kind"),
    [
        (1, 0.94, 11, rules.REASON_EXIT_STOP, "exit_full"),
        (1, 1.10, 15, rules.REASON_ADD_STEP, "add_step"),
        (1, 1.00, 15, rules.REASON_EXIT_TIME, "exit_full"),
        (2, 1.03, 21, rules.REASON_EXIT_STOP, "exit_full"),
        (2, 1.20, 25, rules.REASON_REDUCE_TIER, "reduce_step"),
        (2, 1.05, 25, rules.REASON_EXIT_TIME, "exit_full"),
        # 1.03A is the hard stop, so it necessarily wins before the
        # lower 1.01A fallback can be reached.
        (2, 1.01, 21, rules.REASON_EXIT_STOP, "exit_full"),
        (3, 1.30, 30, rules.REASON_REDUCE_TIER, "reduce_step"),
        (3, 1.08, 30, rules.REASON_EXIT_TRAIL_STAGE, "exit_full"),
        (4, 1.40, 40, rules.REASON_EXIT_FINAL, "exit_full"),
        (4, 1.15, 40, rules.REASON_EXIT_TRAIL_STAGE, "exit_full"),
    ],
)
def test_state_machine_thresholds_are_inclusive_and_success_wins_on_day_five(
    state, ratio, seq_idx, reason, kind,
):
    pos = _pos(state)
    action = _action(pos, ratio, seq_idx)
    assert action is not None
    assert action[:2] == (reason, kind)


@pytest.mark.parametrize(
    ("state", "ratio", "seq_idx"),
    [
        (1, 1.00, 14),
        (2, 1.05, 24),
        (3, 1.081, 30),
        (4, 1.151, 40),
    ],
)
def test_state_machine_does_not_fire_before_boundary(state, ratio, seq_idx):
    assert _action(_pos(state), ratio, seq_idx) is None


def test_historical_target_flags_disable_only_the_corresponding_fallback():
    assert _action(_pos(2, reached_120=True), 1.01, 30)[:2] == (
        rules.REASON_EXIT_STOP, "exit_full",
    )
    assert _action(_pos(3, reached_130=True), 1.08, 30) is None
    assert _action(_pos(4, reached_140=True), 1.15, 40) is None


def test_executor_runs_full_fixed_base_tier_path_and_records_context():
    cfg = load_config()
    cfg.execution.slippage_bp = 0
    dates = pd.bdate_range("2024-01-01", periods=7)
    closes = [100, 100, 110, 120, 130, 140, 140]
    opens = [100, 100, 100, 110, 120, 130, 140]
    frame = pd.DataFrame({
        "trade_date": dates,
        "symbol": "TEST",
        "open": opens,
        "high": [p + 1 for p in opens],
        "low": [p - 1 for p in opens],
        "close": closes,
        "volume": 1_000_000,
        "entry_candidate": [True, False, False, False, False, False, False],
        "market_regime": ["down"] * 7,
        "stock_trend": ["up"] * 7,
    })

    result = run_executor(frame, cfg)
    fills = result["fills_df"].reset_index(drop=True)
    trade = result["trades_df"].iloc[0]

    assert fills["order_type"].tolist() == [
        "init", "add1", "reduce", "reduce", "exit",
    ]
    # The add tops current marked value up to the 10% single-name cap.
    # Both reductions use the same post-add peak base: floor(913 / 3)=304.
    assert fills["qty"].tolist() == [500, 413, 304, 304, 305]
    assert fills["reason"].tolist() == [
        rules.REASON_ENTRY_PULLBACK_PREDICT,
        rules.REASON_ADD_STEP,
        rules.REASON_REDUCE_TIER,
        rules.REASON_REDUCE_TIER,
        rules.REASON_EXIT_FINAL,
    ]
    assert trade["A"] == pytest.approx(100)
    assert trade["initial_risk_B"] == pytest.approx(3_000)
    assert trade["entry_market_regime"] == "down"
    assert trade["entry_stock_trend"] == "up"


def test_executor_does_not_filter_high_price_stock_on_board_lot_details():
    cfg = load_config()
    cfg.execution.slippage_bp = 0
    dates = pd.bdate_range("2024-01-01", periods=7)
    closes = [300, 300, 330, 360, 390, 420, 420]
    opens = [300, 300, 300, 330, 360, 390, 420]
    frame = pd.DataFrame({
        "trade_date": dates,
        "symbol": "EXPENSIVE",
        "open": opens,
        "high": [p + 1 for p in opens],
        "low": [p - 1 for p in opens],
        "close": closes,
        "volume": 1_000_000,
        "entry_candidate": [True, False, False, False, False, False, False],
        "market_regime": ["sideways"] * 7,
        "stock_trend": ["sideways"] * 7,
    })

    result = run_executor(frame, cfg)

    assert result["fills_df"].iloc[0]["qty"] == 166
    assert result["fills_df"]["order_type"].tolist() == [
        "init", "add1", "reduce", "reduce", "exit",
    ]
    assert result["positions_df"].empty
