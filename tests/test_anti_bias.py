"""防偏差硬性检查:
1. 指标/信号在 symbol 内 shift 界定,绝不引用未来 K 线
2. 所有订单由 T 日信号 -> T+1 日开盘价 产生
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.executor import run_executor
from backtest.vector_research import compute_daily_features
from config_loader import load_config


def _synth_bars(n=400, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    # 带向上漂移 + 噪声,确保出现趋势 + 回调 + hist 收敛
    rets = rng.normal(0.0012, 0.022, n)
    close = 10.0 * np.cumprod(1 + rets)
    open_ = close * (1 + rng.normal(0, 0.004))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006)))
    bars = pd.DataFrame({
        "trade_date": dates, "symbol": "000001",
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(1e6, 1e7, n).astype(float),
        "amount": close * 1e6, "adj_factor": 1.0,
    })
    bench = pd.DataFrame({
        "trade_date": dates, "symbol": "000300",
        "open": close * 1.0, "high": high * 1.0, "low": low * 1.0, "close": close * 1.0,
        "volume": 1e8, "amount": 1e9,
    })
    return bars, bench


def _entry_friendly_cfg():
    cfg = load_config(profile="aggressive")
    cfg.features.trend_score_min = 3
    cfg.signal.prior_pos_bars = 1
    cfg.signal.min_neg_bars = 1
    cfg.signal.max_neg_bars = 10
    cfg.signal.convergence_count = 1
    cfg.signal.min_strength_ratio = 0
    cfg.position.fixed_weight = 0.05
    return cfg


# ---------------- 1. 无未来信息 ----------------
def test_features_no_lookahead():
    bars, bench = _synth_bars()
    cfg = load_config(profile="neutral")
    f1 = compute_daily_features(bars, bench, cfg)
    bars2 = bars.copy()
    bars2.loc[bars2.index[-1], "close"] *= 3.0  # 扰动最后一根(未来)
    f2 = compute_daily_features(bars2, bench, cfg)

    cols = ["ma_fast", "ma_mid", "ma_slow", "dif", "dea", "hist", "atr",
            "vol20", "trend_score", "prior_pos_run", "neg_run_k", "entry_candidate"]
    a = f1.iloc[:-1][cols].reset_index(drop=True)
    b = f2.iloc[:-1][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)  # 扰动未来不应改变任何历史行的指标/信号


# ---------------- 2. T 日信号 -> T+1 开盘 ----------------
def test_orders_t_plus_one():
    bars, bench = _synth_bars()
    cfg = _entry_friendly_cfg()
    feature_df = compute_daily_features(bars, bench, cfg)
    res = run_executor(feature_df, cfg)
    fills = res["fills_df"]
    inits = fills[fills["order_type"] == "init"] if not fills.empty else fills
    if inits.empty:
        pytest.skip("合成数据未产生 init 成交(机制仍由其它测试覆盖)")

    dates = list(feature_df["trade_date"].unique())
    pos = {d: i for i, d in enumerate(dates)}
    for _, row in inits.iterrows():
        i = pos[row["trade_date"]]
        assert i >= 1, "init 成交日不应是首日"
        prev_date = dates[i - 1]
        sig = feature_df[(feature_df.trade_date == prev_date)
                         & (feature_df.symbol == row["symbol"])]["entry_candidate"]
        assert len(sig) > 0 and bool(sig.iloc[0]), \
            f"init 成交 {row['trade_date']} 必须由 T-1({prev_date}) 的 entry_candidate 信号驱动"
