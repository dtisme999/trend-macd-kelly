"""列名归一化:AKShare/Tushare 中文列 -> 统一英文 schema。

统一口径:
- bars_daily: trade_date, symbol, open, high, low, close, volume, amount, adj_factor
- benchmark_daily: trade_date, symbol, open, high, low, close, volume, amount
"""
from __future__ import annotations

import pandas as pd

_STOCK_COL_MAP = {
    "日期": "trade_date",
    "股票代码": "symbol",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}
_INDEX_COL_MAP = {
    "日期": "trade_date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}
_NUMERIC_COLS = ["open", "high", "low", "close", "volume", "amount"]


def normalize_stock_daily(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = raw.rename(columns=_STOCK_COL_MAP)
    keep = ["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if "symbol" not in df.columns or df["symbol"].isna().all():
        df["symbol"] = symbol
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    for c in _NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["adj_factor"] = 1.0  # hfq 已是复权价;保留字段以对齐 schema
    return df.dropna(subset=["open", "high", "low", "close"]).sort_values("trade_date")


def normalize_index_daily(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = raw.rename(columns=_INDEX_COL_MAP)
    keep = ["trade_date", "open", "high", "low", "close", "volume", "amount"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["symbol"] = symbol
    for c in _NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).sort_values("trade_date")
