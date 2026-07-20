"""MACD 收敛信号状态机。

- death_cross: DIF 下穿 DEA
- hist_shrink: hist<0 且 |hist| < |hist_{t-1}|(空头动能收敛)
- conv_n: hist_shrink 连续为真的根数
- bars_since_dc: 距上次 death_cross 的根数(NaN 表示历史无死叉 -> 不入场)
- entry_candidate: trend_score>=min & 死叉后等待 N 根 & conv_n>=k & (可选 hist<0 / DIF 拐头)

策略语义:"趋势股中,等一次死叉(调整)后,空头动能连续收敛时重启入场"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _consecutive_true_count(s: pd.Series) -> pd.Series:
    v = s.astype(int)
    grp = (v.diff().fillna(1) != 0).cumsum()
    cnt = v.groupby(grp).cumsum()
    return cnt.where(v == 1, 0)


def _bars_since_true(s: pd.Series) -> pd.Series:
    """距上次 True 的根数;无历史 True 返回 NaN(使 >= 比较为 False)。"""
    v = s.astype(int).values
    n = len(v)
    pos = np.arange(n)
    last = np.where(v == 1, pos, np.nan).astype(float)
    last = pd.Series(last).ffill().values
    out = np.where(np.isnan(last), np.nan, pos - last)
    return pd.Series(out, index=s.index)


def compute_macd_signals(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """cfg 为完整 AppConfig(需 features.trend_score_min 与 signal.*)。"""
    g = df.groupby("symbol", group_keys=False)

    df["death_cross"] = (
        (df["dif"] < df["dea"])
        & (g["dif"].shift(1) >= g["dea"].shift(1))
    ).astype(int)

    prev_abs_hist = g["hist"].transform(lambda s: s.shift(1).abs())
    df["hist_shrink"] = (
        (df["hist"] < 0) & (df["hist"].abs() < prev_abs_hist)
    ).astype(int)

    df["conv_n"] = g["hist_shrink"].transform(_consecutive_true_count)
    df["bars_since_dc"] = g["death_cross"].transform(_bars_since_true)

    sig = cfg.signal
    entry = (
        (df["trend_score"] >= cfg.features.trend_score_min)
        & df["bars_since_dc"].notna()
        & (df["bars_since_dc"] >= sig.wait_bars_after_death_cross)
        & (df["conv_n"] >= sig.convergence_bars)
    )
    if sig.require_hist_below_zero:
        entry = entry & (df["hist"] < 0)
    if sig.require_dif_turn_up:
        entry = entry & (df["dif"] > g["dif"].shift(1))

    df["entry_candidate"] = entry.fillna(False)
    return df
