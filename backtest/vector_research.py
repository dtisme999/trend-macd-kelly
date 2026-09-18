"""向量化研究层:组装特征 + 信号状态机 + 参数网格扫描。

原则(防 look-ahead):
- 所有指标/信号在 symbol 内部用 shift/rolling/ewm 界定可用时间,绝不引用未来 K 线。
- 本层只产出"日级目标动作"(entry_candidate 等),真实成交由 backtest/executor.py 在 T+1 开盘完成。
"""
from __future__ import annotations

import copy
import itertools

import pandas as pd

from strategy.indicators import compute_indicators
from strategy.macd_convergence import compute_macd_signals
from strategy.trend import compute_trend_score


def compute_daily_features(bars: pd.DataFrame, bench_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """数据 -> 特征+信号(含 entry_candidate)。返回长表,按 symbol/trade_date 排序。"""
    df = compute_indicators(bars, cfg.features)
    df = compute_trend_score(df, bench_df, cfg.features)
    df = compute_macd_signals(df, cfg)
    return df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


SIGNAL_COLUMNS = [
    "trade_date", "symbol", "open", "high", "low", "close", "volume", "amount",
    "ma_fast", "ma_mid", "ma_slow", "dif", "dea", "hist", "atr", "vol20",
    "trend_score", "trend_ok", "stock_trend", "market_regime",
    "prior_pos_run", "neg_run_k", "convergence_count", "prior_pos_max",
    "current_neg_min", "macd_strength_ratio", "entry_candidate",
]


def build_signal_df(feature_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SIGNAL_COLUMNS if c in feature_df.columns]
    return feature_df[cols].copy()


def _set_nested(cfg_obj, dotted: str, value):
    obj = cfg_obj
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def param_grid_scan(base_cfg, param_grid: dict, run_fn, metric_fn) -> pd.DataFrame:
    """参数网格扫描。

    param_grid: {"features.trend_score_min": [3,4,5], "signal.min_strength_ratio": [2,3], ...} (dotted keys)
    run_fn(cfg) -> 结果对象(传给 metric_fn)
    metric_fn(result) -> dict 指标
    返回 runs_df:每行一个参数组合 + 指标。
    """
    keys = list(param_grid.keys())
    runs = []
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    for vals in combos:
        cfg = copy.deepcopy(base_cfg)
        override = {}
        for k, v in zip(keys, vals):
            _set_nested(cfg, k, v)
            override[k] = v
        try:
            result = run_fn(cfg)
            m = metric_fn(result) or {}
        except Exception as e:  # noqa: BLE001
            m = {"error": str(e)}
        runs.append({**override, **m})
    return pd.DataFrame(runs)
