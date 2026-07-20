"""仓位规则与原因标签(纯函数,操作 Position 状态)。

原因标签(MVP 保持精简,便于复盘):
- TREND_OK / MACD_CONVERGING : 入场理由
- ADD_ON_BREAKOUT / ADD_ON_PROFIT : 加仓理由
- EXIT_ON_DRAWDOWN / EXIT_ON_STOP / EXIT_ON_TIME / EXIT_ON_INEFFECTIVE : 退出理由

Position(由 executor 定义,duck-typed)需要属性:
  cost, highest_close, highest_high_since_entry, entry_date,
  atr_at_entry, add1_done, add2_done, reduce1_done, reduce2_done
"""
from __future__ import annotations

REASON_TREND_OK = "TREND_OK"
REASON_MACD_CONVERGING = "MACD_CONVERGING"
REASON_ADD_ON_BREAKOUT = "ADD_ON_BREAKOUT"
REASON_ADD_ON_PROFIT = "ADD_ON_PROFIT"
REASON_EXIT_ON_DRAWDOWN = "EXIT_ON_DRAWDOWN"
REASON_EXIT_ON_STOP = "EXIT_ON_STOP"
REASON_EXIT_ON_TIME = "EXIT_ON_TIME"
REASON_EXIT_ON_INEFFECTIVE = "EXIT_ON_INEFFECTIVE"


def unrealized_return(pos, close: float) -> float:
    return close / pos.cost - 1.0 if pos.cost > 0 else 0.0


def check_add1(pos, close: float, cfg):
    """首次加仓:突破调整高点(默认)或浮盈达 add1_profit_pct。返回 reason 或 None。"""
    if pos.add1_done:
        return None
    if cfg.add1_mode == "profit":
        if unrealized_return(pos, close) >= cfg.add1_profit_pct:
            return REASON_ADD_ON_PROFIT
    else:
        if pos.highest_high_since_entry and close > pos.highest_high_since_entry:
            return REASON_ADD_ON_BREAKOUT
    return None


def check_add2(pos, close: float, cfg):
    """第二次加仓:浮盈 >= add2_profit_atr 倍 ATR(以建仓时 ATR 计)。"""
    if pos.add2_done:
        return None
    if pos.atr_at_entry <= 0:
        return None
    profit_atr = (close - pos.cost) / pos.atr_at_entry
    if profit_atr >= cfg.add2_profit_atr:
        return REASON_ADD_ON_PROFIT
    return None


def check_full_exit(pos, close: float, today_date, cfg):
    """清仓级退出:初始止损 / 回撤清仓 / 持仓超时 / 无效持仓。

    返回 (reason, 1.0) 或 None。对应执行器 use_exits 开关(B2+ 启用)。
    """
    # 1. 初始止损(ATR)
    stop_price = pos.cost - cfg.init_stop_atr * pos.atr_at_entry
    if close < stop_price:
        return REASON_EXIT_ON_STOP, 1.0

    # 2. 回撤清仓
    dd = 1.0 - close / pos.highest_close if pos.highest_close > 0 else 0.0
    if dd >= cfg.drawdown_exit:
        return REASON_EXIT_ON_DRAWDOWN, 1.0

    # 3. 持仓时间
    holding_days = (today_date - pos.entry_date).days
    if holding_days > cfg.max_holding_days:
        return REASON_EXIT_ON_TIME, 1.0
    if (holding_days > cfg.ineffective_holding_days
            and unrealized_return(pos, close) < cfg.min_effective_return):
        return REASON_EXIT_ON_INEFFECTIVE, 1.0
    return None


def check_reduce(pos, close: float, cfg):
    """回撤分档减仓(部分退出)。返回 (reason, fraction) 或 None。

    对应执行器 use_reduces 开关(B3+ 启用)。reduce1_done/reduce2_done 防同档重复。
    """
    dd = 1.0 - close / pos.highest_close if pos.highest_close > 0 else 0.0
    if dd >= cfg.drawdown_reduce_2 and not pos.reduce2_done:
        return REASON_EXIT_ON_DRAWDOWN, cfg.reduce_fraction_2
    if dd >= cfg.drawdown_reduce_1 and not pos.reduce1_done:
        return REASON_EXIT_ON_DRAWDOWN, cfg.reduce_fraction_1
    return None
