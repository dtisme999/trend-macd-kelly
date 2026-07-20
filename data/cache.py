"""Parquet 缓存与合成数据生成。

设计要点:
- "文件即数据库":原始 bars/benchmark/calendar 落 Parquet,可迁移、可重跑。
- 合成数据生成器:当 AKShare 网络不可用时,保证研究管道仍可端到端跑通(仅用于冒烟测试,结果无研究意义)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_path(name: str) -> Path:
    return CACHE_DIR / name


def exists(name: str) -> bool:
    return cache_path(name).exists()


def write_parquet(df: pd.DataFrame, name: str) -> Path:
    path = cache_path(name)
    df.to_parquet(path, index=False)
    return path


def read_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(cache_path(name))


# --------------------------------------------------------------------------- #
# 合成数据(兜底,仅用于管道冒烟测试)
# --------------------------------------------------------------------------- #
def generate_synthetic_bars(
    symbols: list[str],
    calendar_dates: list[pd.Timestamp],
    seed: int = 42,
) -> pd.DataFrame:
    """生成 N 只股票的合成日线(带共同市场因子 + 个股 beta/drift/vol)。"""
    rng = np.random.default_rng(seed)
    n = len(calendar_dates)
    # 共同市场因子
    mkt_ret = rng.normal(0.0003, 0.012, n)
    mkt_ret[0] = 0.0
    mkt_px = np.cumprod(1.0 + mkt_ret)

    rows = []
    for sym in symbols:
        beta = rng.uniform(0.6, 1.4)
        drift = rng.normal(0.0004, 0.0003)
        vol = rng.uniform(0.012, 0.03)
        rets = drift + beta * (mkt_ret) * 0.3 + rng.normal(0.0, vol, n)
        px = 10.0 * np.cumprod(1.0 + rets)
        for i, dt in enumerate(calendar_dates):
            close = float(px[i])
            open_ = close * (1.0 + rng.normal(0.0, 0.005))
            hi = max(open_, close) * (1.0 + abs(rng.normal(0.0, 0.006)))
            lo = min(open_, close) * (1.0 - abs(rng.normal(0.0, 0.006)))
            volume = int(rng.integers(500_000, 5_000_000))
            amount = volume * close
            rows.append((dt, sym, open_, hi, lo, close, volume, amount, 1.0))

    return pd.DataFrame(
        rows,
        columns=[
            "trade_date", "symbol", "open", "high", "low",
            "close", "volume", "amount", "adj_factor",
        ],
    )


def generate_synthetic_benchmark(
    symbol: str,
    calendar_dates: list[pd.Timestamp],
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    n = len(calendar_dates)
    rets = rng.normal(0.0003, 0.010, n)
    rets[0] = 0.0
    px = 3000.0 * np.cumprod(1.0 + rets)
    rows = []
    for i, dt in enumerate(calendar_dates):
        close = float(px[i])
        open_ = close * (1.0 + rng.normal(0.0, 0.004))
        hi = max(open_, close) * (1.0 + abs(rng.normal(0.0, 0.005)))
        lo = min(open_, close) * (1.0 - abs(rng.normal(0.0, 0.005)))
        rows.append((dt, symbol, open_, hi, lo, close, int(rng.integers(1e8, 1e10)), close * 1e8))
    return pd.DataFrame(
        rows,
        columns=["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"],
    )
