"""趋势分(trend_score):6 个布尔条件求和。

条件:
1. close > ma_mid(中期均线上方)
2. ma_fast > ma_mid(短期在上)
3. ma_mid > ma_slow(中期均线抬升排列)
4. ma_mid 20 日变化 > 0(中期均线斜率向上)
5. close 60 日变化 > 0(中期价格向上)
6. 相对强度 rs = close/bench_close 高于其 60 日均值(跑赢基准近期)
trend_ok = trend_score >= cfg.trend_score_min
"""
from __future__ import annotations

import pandas as pd


def compute_trend_score(df: pd.DataFrame, bench_df: pd.DataFrame, cfg) -> pd.DataFrame:
    bc = bench_df[["trade_date", "close"]].rename(columns={"close": "bench_close"})
    df = df.merge(bc, on="trade_date", how="left")
    df["bench_close"] = df.groupby("symbol")["bench_close"].ffill()
    df["rs"] = df["close"] / df["bench_close"]
    df["rs_ma60"] = df.groupby("symbol", group_keys=False)["rs"].transform(
        lambda s: s.rolling(60).mean()
    )

    g = df.groupby("symbol", group_keys=False)
    c1 = (df["close"] > df["ma_mid"]).astype(int)
    c2 = (df["ma_fast"] > df["ma_mid"]).astype(int)
    c3 = (df["ma_mid"] > df["ma_slow"]).astype(int)
    c4 = (g["ma_mid"].transform(lambda s: s.pct_change(20)) > 0).astype(int)
    c5 = (g["close"].transform(lambda s: s.pct_change(60)) > 0).astype(int)
    c6 = (df["rs"] > df["rs_ma60"]).astype(int)

    df["trend_score"] = c1 + c2 + c3 + c4 + c5 + c6
    df["trend_ok"] = df["trend_score"] >= cfg.trend_score_min
    return df
