# Exact MACD Regime State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the screening, three-way regime comparison, and A-anchored position state machine exactly match the approved requirements and prove it with automated and cached-market-data tests.

**Architecture:** Keep feature creation vectorized by symbol, add a focused regime/performance module, and keep trading decisions in the pure `strategy.rules.check_state_action` function. The event executor remains responsible only for T+1 order execution, lots, fees, and recording entry-time labels.

**Tech Stack:** Python 3.13, pandas, NumPy, pydantic, pytest.

## Global Constraints

- MACD histogram is `2 * (DIF - DEA)`.
- Entry thresholds are negative run 2–7, first convergence count exactly 1, prior positive run at least 7, and strength ratio at least 2.
- Price triggers use current close and execute at next tradable open.
- Threshold multipliers are 0.94, 1.01, 1.03, 1.10, 1.20, 1.30, and 1.40 of first fill price A.
- Market and stock labels are exactly `up`, `down`, and `sideways`.

---

### Task 1: Exact MACD screening

**Files:**
- Modify: `config/default.yaml`
- Modify: `config/profiles.yaml`
- Modify: `config_loader.py`
- Modify: `strategy/macd_convergence.py`
- Test: `tests/test_macd_screening.py`

**Interfaces:**
- Consumes: a symbol-sorted DataFrame with `hist`.
- Produces: `prior_pos_run`, `neg_run_k`, `convergence_count`, `prior_pos_max`, `current_neg_min`, `macd_strength_ratio`, and `entry_candidate`.

- [ ] Write table-driven failing tests for runs 1/2/7/8, convergence count 0/1/2, prior run 6/7, strength ratio below/equal to 2, and a future-row perturbation.
- [ ] Run `.venv/bin/python -m pytest tests/test_macd_screening.py -v` and verify requirement assertions fail against the old implementation.
- [ ] Implement causal segment statistics and the exact conjunction without `trend_score`.
- [ ] Run the focused tests and then the existing anti-bias tests.

### Task 2: Three-way regime classification and performance

**Files:**
- Create: `backtest/regime.py`
- Modify: `strategy/trend.py`
- Modify: `backtest/vector_research.py`
- Modify: `run_backtest.py`
- Test: `tests/test_regime.py`

**Interfaces:**
- Produces: feature columns `stock_trend`, `market_regime`.
- Produces: `summarize_market_regime(equity_df, feature_df)` and `summarize_stock_trend(trades_df)`.

- [ ] Write failing deterministic up/down/sideways classification tests and aggregation-recalculation tests.
- [ ] Run the focused test and verify missing interfaces fail.
- [ ] Add causal MA20/MA60 classification, merge benchmark labels, and implement performance summaries.
- [ ] Print and persist the two summaries in the CLI path.

### Task 3: Exact position state machine

**Files:**
- Modify: `strategy/rules.py`
- Modify: `backtest/executor.py`
- Test: `tests/test_position_state_machine.py`

**Interfaces:**
- `check_state_action(pos, close, seq_idx, cfg_risk)` remains the pure decision API.
- `Position` records `entry_market_regime`, `entry_stock_trend`, `peak_qty`, and `initial_risk_B`.

- [ ] Write failing tests for every threshold equality, timeout day 4/5, action priority, and state transition.
- [ ] Run the focused tests and confirm behavior failures rather than fixture errors.
- [ ] Change inclusive thresholds and fixed-base reductions; record B and entry labels.
- [ ] Run pure and executor integration tests, confirming T+1 execution remains intact.

### Task 4: End-to-end verification and documentation

**Files:**
- Modify: `README.md`
- Modify: `app/streamlit_app.py` only if cached result presentation requires it.

**Interfaces:**
- CLI prints exact signal count, ordinary metrics, and both comparison tables.

- [ ] Run `.venv/bin/python -m pytest tests/ -v`.
- [ ] Run `.venv/bin/python run_backtest.py --skip-validation --universe-size 20` against the existing cache.
- [ ] Inspect fills for invalid state sequences, duplicate tier reductions, quantity errors, and threshold reason coverage.
- [ ] Re-read the approved spec line by line and report any remaining limitation explicitly.

