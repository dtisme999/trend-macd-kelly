"""技术指标计算(统一口径,shift 安全,无未来信息)。

固定口径(报告强调"从第一天就固定"):
- MA: 简单移动平均
- MACD: DIF = EMA(fast) - EMA(slow); DEA = EMA(signal, DIF); hist = 2*(DIF-DEA)
- EMA: adjust=False(与国内常用 MACD 一致)
- ATR: Wilder 平滑(RMA, alpha=1/period,等价 TA-Lib ATR)
- vol: close 日收益的滚动标准差
所有指标均按 symbol 分组,使用 shift/rolling/ewm,绝不引用未来 K 线。
"""
from __future__ import annotations

import pandas as pd


def compute_indicators(bars: pd.DataFrame, cfg) -> pd.DataFrame:
    df = bars.sort_values(["symbol", "trade_date"]).copy()

    # True Range(按 symbol shift 前收)
    prev_close = df.groupby("symbol")["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["_tr"] = tr

    g = df.groupby("symbol", group_keys=False)
    df["ma_fast"] = g["close"].transform(lambda s: s.rolling(cfg.ma_fast).mean())
    df["ma_mid"] = g["close"].transform(lambda s: s.rolling(cfg.ma_mid).mean())
    df["ma_slow"] = g["close"].transform(lambda s: s.rolling(cfg.ma_slow).mean())

    df["dif"] = g["close"].transform(
        lambda s: s.ewm(span=cfg.macd_fast, adjust=False).mean()
        - s.ewm(span=cfg.macd_slow, adjust=False).mean()
    )
    df["dea"] = df.groupby("symbol", group_keys=False)["dif"].transform(
        lambda s: s.ewm(span=cfg.macd_signal, adjust=False).mean()
    )
    df["hist"] = 2.0 * (df["dif"] - df["dea"])

    df["atr"] = df.groupby("symbol", group_keys=False)["_tr"].transform(
        lambda s: s.ewm(alpha=1.0 / cfg.atr_period, adjust=False).mean()
    )
    df["vol20"] = df.groupby("symbol", group_keys=False)["close"].transform(
        lambda s: s.pct_change().rolling(cfg.vol_period).std()
    )
    return df.drop(columns=["_tr"])
