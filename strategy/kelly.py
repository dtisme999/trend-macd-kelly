"""简化凯利仓位器(1/4 Kelly + 样本置信收缩 + 波动调整 + 上下限)。

四步法(报告 "凯利仓位的简化实现"):
1. 样本池:仅用"已闭合"的历史交易(exit < 当前日期),禁止看未来
2. 估计:p̂=(W+1)/(N+2) 拉普拉斯平滑;b̂=mean(win)/|mean(loss)|
3. 原始凯利 f_raw=max((p̂b̂-(1-p̂))/b̂,0),乘样本置信 conf(N) 收缩
4. 落地:w=clip(f_raw * kelly_fraction * conf(N) * vol_adj, w_min, w_max)
   样本不足或无优势时退回 fallback_fixed_weight(便于与固定仓位对照组比较)
"""
from __future__ import annotations

import numpy as np


def estimate_p_hat(wins: int, n: int) -> float:
    """拉普拉斯平滑胜率,避免小样本 0/1。"""
    return (wins + 1) / (n + 2)


def estimate_b_hat(returns) -> float:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return np.nan
    return float(wins.mean() / abs(losses.mean()))


def raw_kelly(p: float, b: float) -> float:
    if not np.isfinite(b) or b <= 0:
        return 0.0
    return max((p * b - (1 - p)) / b, 0.0)


def conf_n(n: int) -> float:
    """样本置信收缩系数。"""
    if n < 20:
        return 0.0
    if n < 40:
        return 0.35
    if n < 80:
        return 0.60
    if n < 150:
        return 0.80
    return 1.0


def vol_adj(sigma20: float, sigma_star: float) -> float:
    """高波折扣、低波略加,clip 到 [0.7, 1.1]。"""
    if not np.isfinite(sigma20) or sigma20 <= 0:
        return 1.0
    return float(np.clip(sigma_star / sigma20, 0.7, 1.1))


def kelly_weight(closed_returns, sigma20: float, cfg) -> float:
    """
    closed_returns: 已闭合交易收益率 Series(调用方须保证仅含 exit<当前日期 的交易)
    sigma20: 当日 20 日波动率
    返回最终权重(占权益比例)
    """
    n = len(closed_returns)
    if n < cfg.kelly_min_samples:
        return cfg.fallback_fixed_weight

    rets = closed_returns.astype(float)
    wins = int((rets > 0).sum())
    p = estimate_p_hat(wins, n)
    b = estimate_b_hat(rets)
    if not np.isfinite(b) or b <= 0:
        return cfg.fallback_fixed_weight

    f_raw = raw_kelly(p, b)
    w = f_raw * cfg.kelly_fraction * conf_n(n) * vol_adj(sigma20, cfg.vol_target)
    if not np.isfinite(w) or w <= 0:
        return cfg.fallback_fixed_weight
    return float(np.clip(w, cfg.kelly_weight_min, cfg.kelly_weight_max))
