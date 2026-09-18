"""交易状态机(以首笔成交价 A 为锚点,收盘价触发,T+1 开盘成交)。

状态含义
--------
S1(仅首仓 5%):
  - close < (1 - stop_s1_pct) * A            -> 全清止损        (EXIT_STOP)
  - 建仓后 timeout_days 交易日内未曾 close ≥ (1+add_trigger_pct)*A -> 全清超时 (EXIT_TIME)
  - close ≥ (1 + add_trigger_pct) * A        -> 加 5%           (ADD_STEP,状态推进到 S2)

S2(首仓 + 一档加仓,合计 10%):
  - close < (1 + stop_s2_pct) * A            -> 全清止损        (EXIT_STOP)
  - 加仓后 timeout_days 交易日内未曾 close ≥ (1+tp1_pct)*A -> 全清超时 (EXIT_TIME)
  - 尚未 reached_120 且 close ≤ (1 + lock_inband_pct)*A -> 全清保本 (EXIT_LOCK_INBAND)
  - close ≥ (1 + tp1_pct) * A                -> 减 1/3           (REDUCE_TIER,-> S3)

S3(6.67% 剩余):
  - 尚未 reached_130 且 close ≤ (1 + trail_s3_pct)*A -> 剩余全清 (EXIT_TRAIL_STAGE)
  - close ≥ (1 + tp2_pct) * A                -> 再减 1/3         (REDUCE_TIER,-> S4)

S4(3.33% 剩余):
  - 尚未 reached_140 且 close ≤ (1 + trail_s4_pct)*A -> 剩余全清 (EXIT_TRAIL_STAGE)
  - close ≥ (1 + tp3_pct) * A                -> 全清             (EXIT_FINAL)

要求 Position(duck-typed)提供:
  A, state, entry_seq_idx, add_seq_idx,
  reached_120, reached_130, reached_140,
  qty (>0)
"""
from __future__ import annotations

from typing import Optional, Tuple

# ---- 原因标签(供 fills.reason / trades 分析) ---------------------------- #
REASON_ENTRY_PULLBACK_PREDICT = "ENTRY_PULLBACK_PREDICT"
REASON_ADD_STEP = "ADD_STEP"
REASON_REDUCE_TIER = "REDUCE_TIER"
REASON_EXIT_STOP = "EXIT_STOP"
REASON_EXIT_TIME = "EXIT_TIME"
REASON_EXIT_LOCK_INBAND = "EXIT_LOCK_INBAND"
REASON_EXIT_TRAIL_STAGE = "EXIT_TRAIL_STAGE"
REASON_EXIT_FINAL = "EXIT_FINAL"

Action = Tuple[str, str, float]   # (reason, kind, fraction)  kind ∈ {exit_full, reduce_step, add_step}


def check_state_action(pos, close: float, seq_idx: int, cfg_risk) -> Optional[Action]:
    """按状态机判定当日该股需要挂的下一张单;返回 None 表示无动作。

    seq_idx 是当前交易日在该 symbol 时间轴上的 0-based 序号。
    """
    A = pos.A
    if A <= 0 or pos.qty <= 0:
        return None
    r = cfg_risk
    state = pos.state
    ratio = close / A  # close 相对锚点的比值

    # S1
    if state == 1:
        if ratio <= 1.0 - r.stop_s1_pct:
            return (REASON_EXIT_STOP, "exit_full", 1.0)
        # 达标动作优先于同日超时；第 5 个交易日收盘达标仍算成功。
        if ratio >= 1.0 + r.add_trigger_pct:
            return (REASON_ADD_STEP, "add_step", 0.0)
        # 超时:entry_seq_idx 是建仓当日索引,T+1 ~ T+timeout_days 内
        days_since = seq_idx - pos.entry_seq_idx
        if days_since >= r.timeout_days and not pos.reached_110:
            return (REASON_EXIT_TIME, "exit_full", 1.0)
        return None

    # S2
    if state == 2:
        if ratio <= 1.0 + r.stop_s2_pct:
            return (REASON_EXIT_STOP, "exit_full", 1.0)
        if ratio >= 1.0 + r.tp1_pct:
            return (REASON_REDUCE_TIER, "reduce_step", r.reduce_fraction_tier)
        days_since = seq_idx - pos.add_seq_idx if pos.add_seq_idx >= 0 else 0
        if days_since >= r.timeout_days and not pos.reached_120:
            return (REASON_EXIT_TIME, "exit_full", 1.0)
        if (not pos.reached_120) and ratio <= 1.0 + r.lock_inband_pct:
            return (REASON_EXIT_LOCK_INBAND, "exit_full", 1.0)
        return None

    # S3
    if state == 3:
        if ratio >= 1.0 + r.tp2_pct:
            return (REASON_REDUCE_TIER, "reduce_step", 0.5)
        if (not pos.reached_130) and ratio <= 1.0 + r.trail_s3_pct:
            return (REASON_EXIT_TRAIL_STAGE, "exit_full", 1.0)
        return None

    # S4
    if state == 4:
        if ratio >= 1.0 + r.tp3_pct:
            return (REASON_EXIT_FINAL, "exit_full", 1.0)
        if (not pos.reached_140) and ratio <= 1.0 + r.trail_s4_pct:
            return (REASON_EXIT_TRAIL_STAGE, "exit_full", 1.0)
        return None

    return None
