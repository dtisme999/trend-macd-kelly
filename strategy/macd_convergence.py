"""严格、因果的 MACD 柱段筛选。

入场日必须同时满足：
1. 当前 hist < 0，当前连续负柱数为 2..7；
2. 当前是连续收敛（|hist| 递减）的第 1 根；
3. 紧邻前一连续正柱段不少于 7 根；
4. 前正柱段最大值 / abs(当前负柱段截至当日最小值) >= 2。

趋势分类不参与筛选，只用于事后分组对比。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_symbol(sub: pd.DataFrame) -> pd.DataFrame:
    hist = sub["hist"].to_numpy(dtype=float)
    n = len(hist)

    prior_pos_run = np.zeros(n, dtype=int)
    neg_run_k = np.zeros(n, dtype=int)
    convergence_count = np.zeros(n, dtype=int)
    prior_pos_max = np.full(n, np.nan)
    current_neg_min = np.full(n, np.nan)

    positive_run = 0
    positive_max = np.nan
    captured_positive_run = 0
    captured_positive_max = np.nan
    negative_run = 0
    negative_min = np.nan
    shrink_run = 0
    previous_sign = 0
    previous_hist = np.nan

    for i, value in enumerate(hist):
        if np.isnan(value) or value == 0:
            positive_run = 0
            positive_max = np.nan
            captured_positive_run = 0
            captured_positive_max = np.nan
            negative_run = 0
            negative_min = np.nan
            shrink_run = 0
            previous_sign = 0
            previous_hist = value
            continue

        if value > 0:
            if previous_sign == 1:
                positive_run += 1
                positive_max = max(positive_max, value)
            else:
                positive_run = 1
                positive_max = value
            captured_positive_run = 0
            captured_positive_max = np.nan
            negative_run = 0
            negative_min = np.nan
            shrink_run = 0
            previous_sign = 1
            previous_hist = value
            continue

        if previous_sign == 1:
            captured_positive_run = positive_run
            captured_positive_max = positive_max
            negative_run = 1
            negative_min = value
        elif previous_sign == -1:
            negative_run += 1
            negative_min = min(negative_min, value)
        else:
            captured_positive_run = 0
            captured_positive_max = np.nan
            negative_run = 1
            negative_min = value

        if previous_sign == -1 and abs(value) < abs(previous_hist):
            shrink_run += 1
        else:
            shrink_run = 0

        prior_pos_run[i] = captured_positive_run
        neg_run_k[i] = negative_run
        convergence_count[i] = shrink_run
        prior_pos_max[i] = captured_positive_max
        current_neg_min[i] = negative_min
        previous_sign = -1
        previous_hist = value

    denominator = np.abs(current_neg_min)
    strength_ratio = np.divide(
        prior_pos_max,
        denominator,
        out=np.full(n, np.nan),
        where=np.isfinite(prior_pos_max) & np.isfinite(denominator) & (denominator > 0),
    )
    return pd.DataFrame({
        "prior_pos_run": prior_pos_run,
        "neg_run_k": neg_run_k,
        "convergence_count": convergence_count,
        "prior_pos_max": prior_pos_max,
        "current_neg_min": current_neg_min,
        "macd_strength_ratio": strength_ratio,
    }, index=sub.index)


def compute_macd_signals(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """计算严格筛选字段；每只股票必须已按时间排序。"""
    parts = [_compute_symbol(sub) for _, sub in df.groupby("symbol", sort=False)]
    signals = pd.concat(parts).sort_index() if parts else pd.DataFrame(index=df.index)
    out = df.copy()
    for column in signals.columns:
        out[column] = signals[column]

    sig = cfg.signal
    entry = (
        (out["hist"] < 0)
        & (out["neg_run_k"] >= int(sig.min_neg_bars))
        & (out["neg_run_k"] <= int(sig.max_neg_bars))
        & (out["convergence_count"] == int(sig.convergence_count))
        & (out["prior_pos_run"] >= int(sig.prior_pos_bars))
        & (out["macd_strength_ratio"] >= float(sig.min_strength_ratio))
    )
    out["entry_candidate"] = entry.fillna(False).astype(bool)
    return out
