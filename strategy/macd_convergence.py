"""MACD 预判买点信号(回调再涨,提前抄底)。

策略语义:"多头段(hist>0 持续 ≥ N 根)后出现短暂回调(hist<0),
回调段内 |hist| 连续收敛(严格递减) → 在 hist 尚未翻正之前提前买入"。

输出字段:
- prior_pos_run  : 当前 hist<0 段紧邻的上一 hist>0 段长度(在负段内每一行都取该值)
- neg_run_k      : hist<0 段内位置(1=段首,2=第二根,...);hist≥0 时为 0
- entry_candidate: 综合 trend_score + 上述三条件的最终入场信号(bool)

字段对下游 executor/tests 稳定。旧的 death_cross/hist_shrink/conv_n/bars_since_dc
一并删除;若外部代码仍引用会报错,请同步升级。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_symbol(sub: pd.DataFrame, prior_pos_bars: int, min_neg_bars: int,
                    max_neg_bars: int, converge_bars: int) -> pd.DataFrame:
    """针对单只 symbol 计算 prior_pos_run / neg_run_k / conv_ok。"""
    hist = sub["hist"].values
    n = len(hist)
    # 用 -1/+1 分段;hist == 0 视为与前一根同号(避免多余分段)
    sign = np.where(hist > 0, 1, np.where(hist < 0, -1, 0))
    for i in range(1, n):
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    # 首行若为 0,记为 -1(视作负段),不影响后续统计(前 warmup 段本就不发信号)
    if n > 0 and sign[0] == 0:
        sign[0] = -1

    # seg_id:同号连续段唯一编号
    seg_id = np.zeros(n, dtype=int)
    for i in range(1, n):
        seg_id[i] = seg_id[i - 1] + (1 if sign[i] != sign[i - 1] else 0)

    # 每行在其段内的位置(1-indexed)
    k_in_seg = np.zeros(n, dtype=int)
    cur = 0
    prev_seg = -1
    for i in range(n):
        if seg_id[i] != prev_seg:
            cur = 1
            prev_seg = seg_id[i]
        else:
            cur += 1
        k_in_seg[i] = cur

    # 每段总长度
    seg_len = pd.Series(seg_id).groupby(seg_id).transform("size").values

    # 段末长度(仅在段末行有值,其余 NaN)—— 用来给"下一段"取用
    last_pos_in_seg = (np.diff(seg_id, append=seg_id[-1] + 1) != 0)  # 段末为 True
    prev_seg_end_len = np.full(n, np.nan)
    prev_seg_end_len[last_pos_in_seg] = seg_len[last_pos_in_seg]
    # 下一段每一行都取上一段的段长
    prior_pos_run = np.full(n, 0.0)
    last_end_len = np.nan
    last_end_sign = 0
    for i in range(n):
        # 段起点上一行才是上一段段末
        if i > 0 and seg_id[i] != seg_id[i - 1]:
            last_end_len = seg_len[i - 1]
            last_end_sign = sign[i - 1]
        # 只有当前是 hist<0 段,且上一段是 hist>0 段时,才写入 prior_pos_run
        if sign[i] < 0 and last_end_sign > 0 and not np.isnan(last_end_len):
            prior_pos_run[i] = last_end_len

    neg_run_k = np.where(sign < 0, k_in_seg, 0)

    # |hist| 是否连续 converge_bars 根严格递减(含当日)
    abs_hist = np.abs(hist)
    if converge_bars <= 1:
        conv_ok = np.ones(n, dtype=bool)
    else:
        diffs = np.diff(abs_hist)                 # 长度 n-1: abs[i+1]-abs[i]
        strict_down = np.concatenate(([False], diffs < 0))  # 对齐到 [1..n-1]
        # 要求当前及往前 converge_bars-1 根都是 strict_down = True(除首根外)
        need = converge_bars - 1
        conv_ok = np.zeros(n, dtype=bool)
        run = 0
        for i in range(n):
            if strict_down[i]:
                run += 1
            else:
                run = 0
            conv_ok[i] = (run >= need)

    return pd.DataFrame({
        "prior_pos_run": prior_pos_run,
        "neg_run_k": neg_run_k,
        "conv_ok": conv_ok,
    }, index=sub.index)


def compute_macd_signals(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """cfg 为完整 AppConfig(读 features.trend_score_min 与 signal.*)。"""
    sig = cfg.signal
    prior_pos_bars = int(sig.prior_pos_bars)
    min_neg_bars = int(sig.min_neg_bars)
    max_neg_bars = int(sig.max_neg_bars)
    converge_bars = int(sig.converge_bars)

    parts = []
    for sym, sub in df.groupby("symbol", sort=False):
        parts.append(_compute_symbol(sub, prior_pos_bars, min_neg_bars,
                                     max_neg_bars, converge_bars))
    signals = pd.concat(parts).sort_index()
    df = df.copy()
    df["prior_pos_run"] = signals["prior_pos_run"].astype(float)
    df["neg_run_k"] = signals["neg_run_k"].astype(int)
    conv_ok = signals["conv_ok"].astype(bool)

    entry = (
        (df["trend_score"] >= cfg.features.trend_score_min)
        & (df["hist"] < 0)
        & (df["neg_run_k"] >= min_neg_bars)
        & (df["neg_run_k"] <= max_neg_bars)
        & (df["prior_pos_run"] >= prior_pos_bars)
        & conv_ok
    )
    df["entry_candidate"] = entry.fillna(False).astype(bool)
    return df
