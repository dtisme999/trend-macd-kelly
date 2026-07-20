"""轻量事件执行层:按交易日循环,T 日收盘信号 -> T+1 日开盘成交。

天然支持:T+1 开盘成交、手续费、滑点、现金约束、100 股取整、单股/总仓位上限、
分批加仓(突破调整高点/浮盈 ATR)、回撤分档减仓、初始止损(ATR)、最大持仓、
无效持仓退出。FIFO 分批跟踪用于首仓/加仓贡献归因。

防偏差三条硬性规则(对应 tests/test_anti_bias.py):
1. 指标/信号在 symbol 内 shift 界定(在 vector_research 完成)
2. 所有订单由 "T 日信号 -> T+1 日开盘价" 产生(本文件 pending 机制)
3. 凯利 p̂/b̂ 仅来自此前已闭合交易(closed_trades 在卖出成交后才追加)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy import kelly as kelly_mod
from strategy import rules as rules_mod


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Lot:
    lot_type: str          # init | add1 | add2
    qty: int
    price: float
    date: pd.Timestamp


@dataclass
class Order:
    symbol: str
    side: str              # buy | sell
    order_type: str        # init | add1 | add2 | reduce | exit
    target_value: float = 0.0
    reduce_fraction: float = 0.0
    reason: str = ""
    weight: float = 0.0    # 仅 init 用:记录建仓时权重(加仓复用)


@dataclass
class Position:
    symbol: str
    lots: list = field(default_factory=list)
    highest_close: float = 0.0
    highest_high_since_entry: float = 0.0
    entry_date: pd.Timestamp = None
    atr_at_entry: float = 0.0
    add1_done: bool = False
    add2_done: bool = False
    reduce1_done: bool = False
    reduce2_done: bool = False
    entry_trend_score: int = 0
    entry_vol20: float = 0.0
    last_close: float = 0.0
    weight: float = 0.0
    cost_total: float = 0.0
    cost_by_type: dict = field(default_factory=lambda: {"init": 0.0, "add1": 0.0, "add2": 0.0})
    realized_by_type: dict = field(default_factory=lambda: {"init": 0.0, "add1": 0.0, "add2": 0.0})
    realized_pnl_total: float = 0.0
    reduce_fills: list = field(default_factory=list)
    exit_price: float = 0.0

    @property
    def qty(self) -> int:
        return sum(l.qty for l in self.lots)

    @property
    def cost(self) -> float:
        """剩余分批的加权均价(用于止损/浮盈判断)。"""
        q = self.qty
        if q <= 0:
            return 0.0
        return sum(l.price * l.qty for l in self.lots) / q


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _is_tradable(row) -> bool:
    vol = row.get("volume", 0)
    if pd.isna(vol) or vol <= 0:
        return False
    if row["high"] == row["low"]:        # 一字板近似:无法成交
        return False
    return True


def _merge_benchmark(equity_df: pd.DataFrame, bench_df: pd.DataFrame, init_cash: float) -> pd.DataFrame:
    bc = bench_df[["trade_date", "close"]].rename(columns={"close": "bench_close"})
    df = equity_df.merge(bc, on="trade_date", how="left")
    df["bench_close"] = df["bench_close"].ffill().bfill()
    first = df["bench_close"].iloc[0] if len(df) else np.nan
    df["benchmark_equity"] = init_cash * df["bench_close"] / first if first and first > 0 else init_cash
    return df.drop(columns=["bench_close"])


# --------------------------------------------------------------------------- #
# 执行器
# --------------------------------------------------------------------------- #
class Executor:
    def __init__(self, feature_df, cfg, use_macd=True, use_exits=True,
                 use_adds=True, use_reduces=True,
                 position_mode=None, frozen_kelly_returns=None, seed=42):
        self.feature_df = feature_df
        self.cfg = cfg
        self.use_macd = use_macd
        self.use_exits = use_exits
        self.use_adds = use_adds
        self.use_reduces = use_reduces
        self.position_mode = position_mode or cfg.position.mode
        self.frozen_kelly_returns = frozen_kelly_returns
        self.cash = cfg.execution.init_cash
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[dict] = []
        self.fills: list[dict] = []
        self.equity_rows: list[dict] = []
        self.pending: list[Order] = []
        self.last_exit_date: dict = {}
        self._date_groups = {dt: g.set_index("symbol")
                             for dt, g in feature_df.groupby("trade_date")}
        self.trading_dates = sorted(self._date_groups.keys())

    # ---- 主循环 ----
    def run(self):
        for dt in self.trading_dates:
            self._process_day(dt)
        return self._results()

    def _process_day(self, dt):
        today = self._date_groups[dt]
        # 1. 执行昨日收盘生成、今日开盘成交的订单
        for od in self.pending:
            self._execute_order(od, today, dt)
        self.pending = []
        # 2. 盯盘(更新 highest_close,含今日)
        self._mark_to_market(today)
        equity = self._equity()
        # 3. 持仓管理:退出/减仓/加仓(信号今日收盘 -> 明日开盘)
        for sym in list(self.positions.keys()):
            if sym in today.index:
                self._manage_position(sym, today, dt, equity)
        # 4. 新建仓(趋势+MACD收敛信号)
        if self.use_macd:
            self._generate_entries(today, dt, equity)
        # 5. 记录净值
        self._record_equity(dt)

    # ---- 盯盘 ----
    def _mark_to_market(self, today):
        for sym, pos in self.positions.items():
            if sym in today.index:
                close = float(today.loc[sym, "close"])
                pos.last_close = close
                pos.highest_close = max(pos.highest_close, close)

    def _equity(self) -> float:
        mv = sum(p.qty * p.last_close for p in self.positions.values())
        return self.cash + mv

    def _record_equity(self, dt):
        mv = sum(p.qty * p.last_close for p in self.positions.values())
        equity = self.cash + mv
        self.equity_rows.append({
            "trade_date": dt, "equity": equity, "cash": self.cash,
            "market_value": mv, "n_positions": len(self.positions),
            "exposure": (mv / equity) if equity > 0 else 0.0,
        })

    # ---- 订单执行 ----
    def _execute_order(self, od: Order, today, dt):
        if od.symbol not in today.index:
            return
        row = today.loc[od.symbol]
        if not _is_tradable(row):
            return
        open_px = float(row["open"])
        if od.side == "buy":
            self._do_buy(od, open_px, today, dt)
        else:
            self._do_sell(od, open_px, dt)

    def _do_buy(self, od: Order, open_px: float, today, dt):
        ec = self.cfg.execution
        pc = self.cfg.position
        fill_px = open_px * (1 + ec.slippage_bp / 1e4)
        lot = ec.lot_size
        equity = self._equity()

        def cap_qty(q):
            q = int(np.floor(q / lot) * lot)
            return q if q >= lot else 0

        qty = cap_qty(od.target_value / fill_px)
        if qty <= 0:
            return

        # 现金约束
        cost = qty * fill_px
        fee = max(cost * ec.buy_commission_bp / 1e4, ec.min_commission)
        if cost + fee > self.cash:
            qty = cap_qty((self.cash - fee) / fill_px)
            if qty <= 0:
                return
            cost, fee = qty * fill_px, max(cost * ec.buy_commission_bp / 1e4, ec.min_commission)

        # 单股上限
        pos = self.positions.get(od.symbol)
        cur_val = (pos.qty * pos.last_close) if pos else 0.0
        if equity > 0 and (cur_val + cost) / equity > pc.max_single_weight:
            qty = cap_qty((equity * pc.max_single_weight - cur_val) / fill_px)
            if qty <= 0:
                return
            cost, fee = qty * fill_px, max(cost * ec.buy_commission_bp / 1e4, ec.min_commission)

        # 总仓位上限
        total_pos = sum(p.qty * p.last_close for p in self.positions.values())
        if equity > 0 and (total_pos + cost) / equity > pc.max_total_exposure:
            qty = cap_qty((equity * pc.max_total_exposure - total_pos) / fill_px)
            if qty <= 0:
                return
            cost, fee = qty * fill_px, max(cost * ec.buy_commission_bp / 1e4, ec.min_commission)

        self.cash -= (cost + fee)
        if pos is None:
            pos = Position(
                symbol=od.symbol, entry_date=dt,
                atr_at_entry=float(today.loc[od.symbol, "atr"]),
                entry_trend_score=int(today.loc[od.symbol, "trend_score"]),
                entry_vol20=float(today.loc[od.symbol, "vol20"]),
                highest_close=float(today.loc[od.symbol, "close"]),
                highest_high_since_entry=float(today.loc[od.symbol, "high"]),
                last_close=float(today.loc[od.symbol, "close"]),
                weight=od.weight,
            )
            self.positions[od.symbol] = pos
        pos.lots.append(Lot(od.order_type, qty, fill_px, dt))
        pos.cost_total += cost
        pos.cost_by_type[od.order_type] += cost
        if od.order_type == "add1":
            pos.add1_done = True
        elif od.order_type == "add2":
            pos.add2_done = True
        self.fills.append({
            "trade_date": dt, "symbol": od.symbol, "action": "BUY",
            "order_type": od.order_type, "lot_type": od.order_type,
            "price": fill_px, "qty": qty, "fee": fee, "stamp": 0.0, "reason": od.reason,
        })

    def _do_sell(self, od: Order, open_px: float, dt):
        pos = self.positions.get(od.symbol)
        if pos is None or pos.qty <= 0:
            return
        ec = self.cfg.execution
        fill_px = open_px * (1 - ec.slippage_bp / 1e4)
        if od.reduce_fraction >= 1.0:
            qty = pos.qty
        else:
            qty = int((pos.qty * od.reduce_fraction) // ec.lot_size * ec.lot_size)
            if qty < ec.lot_size:
                return
        proceeds = qty * fill_px
        fee = max(proceeds * ec.sell_commission_bp / 1e4, ec.min_commission)
        stamp = proceeds * ec.stamp_tax_sell_bp / 1e4
        net = proceeds - fee - stamp

        # FIFO 减仓
        remaining = qty
        realized = 0.0
        while remaining > 0 and pos.lots:
            lot = pos.lots[0]
            take = min(lot.qty, remaining)
            pnl = (fill_px - lot.price) * take
            realized += pnl
            pos.realized_by_type[lot.lot_type] += pnl
            lot.qty -= take
            remaining -= take
            if lot.qty <= 0:
                pos.lots.pop(0)
        pos.realized_pnl_total += realized
        self.cash += net

        if od.order_type == "reduce":
            pos.reduce_fills.append((fill_px, qty))
        self.fills.append({
            "trade_date": dt, "symbol": od.symbol, "action": "SELL",
            "order_type": od.order_type, "lot_type": "",
            "price": fill_px, "qty": qty, "fee": fee, "stamp": stamp, "reason": od.reason,
        })
        if pos.qty <= 0:
            pos.exit_price = fill_px
            self._close_trade(pos, dt)

    def _close_trade(self, pos: Position, exit_date):
        init_pnl = pos.realized_by_type.get("init", 0.0)
        add_pnl = pos.realized_by_type.get("add1", 0.0) + pos.realized_by_type.get("add2", 0.0)
        reduce_prot = sum(max(0.0, p - pos.exit_price) * q for p, q in pos.reduce_fills)
        invested = pos.cost_total
        ret = pos.realized_pnl_total / invested if invested > 0 else 0.0
        self.closed_trades.append({
            "symbol": pos.symbol, "entry_date": pos.entry_date, "exit_date": exit_date,
            "holding_days": (exit_date - pos.entry_date).days,
            "invested_cost": invested, "realized_pnl": pos.realized_pnl_total, "return": ret,
            "entry_trend_score": pos.entry_trend_score, "entry_vol20": pos.entry_vol20,
            "init_pnl": init_pnl, "add_pnl": add_pnl, "reduce_protection": reduce_prot,
            "n_adds": (1 if pos.add1_done else 0) + (1 if pos.add2_done else 0),
            "weight": pos.weight,
        })
        self.last_exit_date[pos.symbol] = exit_date
        del self.positions[pos.symbol]

    # ---- 持仓管理(退出/减仓/加仓) ----
    def _manage_position(self, sym, today, dt, equity):
        pos = self.positions[sym]
        close = float(today.loc[sym, "close"])
        high = float(today.loc[sym, "high"])
        pos.last_close = close

        # 1. 清仓级退出(止损/回撤清仓/超时/无效)
        if self.use_exits:
            action = rules_mod.check_full_exit(pos, close, dt, self.cfg.risk)
            if action:
                reason, frac = action
                self.pending.append(Order(sym, "sell", "exit",
                                          reduce_fraction=frac, reason=reason))
                pos.highest_high_since_entry = max(pos.highest_high_since_entry, high)
                return

        # 2. 回撤分档减仓(部分)
        if self.use_reduces:
            action = rules_mod.check_reduce(pos, close, self.cfg.risk)
            if action:
                reason, frac = action
                if frac >= self.cfg.risk.reduce_fraction_2:
                    pos.reduce2_done = True
                else:
                    pos.reduce1_done = True
                self.pending.append(Order(sym, "sell", "reduce",
                                          reduce_fraction=frac, reason=reason))
                pos.highest_high_since_entry = max(pos.highest_high_since_entry, high)
                return  # 减仓当日不加仓

        # 3. 加仓(突破调整高点 / 浮盈 ATR)
        if self.use_adds:
            add1_reason = rules_mod.check_add1(pos, close, self.cfg.position)
            if add1_reason:
                target = equity * pos.weight * self.cfg.position.add1_target_ratio - pos.qty * close
                if target > 0:
                    self.pending.append(Order(sym, "buy", "add1",
                                              target_value=target, reason=add1_reason, weight=pos.weight))
            else:
                add2_reason = rules_mod.check_add2(pos, close, self.cfg.position)
                if add2_reason:
                    target = equity * pos.weight * self.cfg.position.add2_target_ratio - pos.qty * close
                    if target > 0:
                        self.pending.append(Order(sym, "buy", "add2",
                                                  target_value=target, reason=add2_reason, weight=pos.weight))
        pos.highest_high_since_entry = max(pos.highest_high_since_entry, high)

    # ---- 新建仓 ----
    def _generate_entries(self, today, dt, equity):
        if "entry_candidate" not in today.columns:
            return
        cand = today.index[today["entry_candidate"].fillna(False)]
        for sym in cand:
            if sym in self.positions:
                continue
            if sym in self.last_exit_date and \
                    (dt - self.last_exit_date[sym]).days < self.cfg.signal.cooldown_days:
                continue
            row = today.loc[sym]
            vol20 = float(row["vol20"])
            weight = self._position_weight(vol20)
            if weight <= 0:
                continue
            target_value = equity * weight * self.cfg.position.init_ratio
            if target_value <= 0:
                continue
            reason = f"{rules_mod.REASON_TREND_OK};{rules_mod.REASON_MACD_CONVERGING}"
            self.pending.append(Order(sym, "buy", "init",
                                      target_value=target_value, reason=reason, weight=weight))

    def _position_weight(self, vol20: float) -> float:
        pc = self.cfg.position
        if self.position_mode == "fixed":
            return pc.fixed_weight
        # fractional_kelly
        if self.frozen_kelly_returns is not None:
            closed_ret = self.frozen_kelly_returns
        else:
            closed_ret = pd.Series([t["return"] for t in self.closed_trades])
        return kelly_mod.kelly_weight(closed_ret, vol20, pc)

    # ---- 结果 ----
    def _results(self):
        equity_df = pd.DataFrame(self.equity_rows)
        if not equity_df.empty:
            equity_df = equity_df.sort_values("trade_date").reset_index(drop=True)
        fills_df = pd.DataFrame(self.fills)
        trades_df = pd.DataFrame(self.closed_trades)
        if self.positions:
            positions_df = pd.DataFrame([{
                "symbol": p.symbol, "qty": p.qty, "cost": p.cost,
                "last_close": p.last_close, "market_value": p.qty * p.last_close,
                "weight": p.weight, "entry_date": p.entry_date,
                "add1_done": p.add1_done, "add2_done": p.add2_done,
            } for p in self.positions.values()])
        else:
            positions_df = pd.DataFrame()
        return {
            "equity_df": equity_df, "fills_df": fills_df,
            "trades_df": trades_df, "positions_df": positions_df,
            "closed_trades": self.closed_trades,
        }


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #
def run_executor(feature_df, cfg, use_macd=True, use_exits=True, use_adds=True,
                 use_reduces=True, position_mode=None, frozen_kelly_returns=None,
                 bench_df=None, seed=42):
    ex = Executor(feature_df, cfg, use_macd=use_macd, use_exits=use_exits,
                  use_adds=use_adds, use_reduces=use_reduces,
                  position_mode=position_mode, frozen_kelly_returns=frozen_kelly_returns, seed=seed)
    res = ex.run()
    if bench_df is not None and not res["equity_df"].empty:
        res["equity_df"] = _merge_benchmark(res["equity_df"], bench_df, cfg.execution.init_cash)
    return res


def run_benchmark_hold(bench_df, init_cash: float) -> pd.DataFrame:
    df = bench_df[["trade_date", "close"]].copy().sort_values("trade_date").reset_index(drop=True)
    first = df["close"].iloc[0]
    df["equity"] = init_cash * df["close"] / first if first > 0 else init_cash
    df["benchmark_equity"] = df["equity"]
    df["cash"] = df["equity"]
    df["market_value"] = 0.0
    df["n_positions"] = 0
    df["exposure"] = 0.0
    return df[["trade_date", "equity", "benchmark_equity", "cash", "market_value", "n_positions", "exposure"]]


def run_equal_weight_trend(feature_df, cfg, bench_df=None, seed=42):
    """对照组 B1:趋势筛选后等权持有,月初再平衡,T+1 开盘成交。"""
    ec = cfg.execution
    dates = sorted(feature_df["trade_date"].unique())
    date_groups = {dt: g.set_index("symbol") for dt, g in feature_df.groupby("trade_date")}
    cash = ec.init_cash
    holdings: dict[str, int] = {}
    last_close: dict[str, float] = {}
    pending = None
    last_month = None
    rows = []

    for dt in dates:
        today = date_groups[dt]
        # 执行再平衡(T-1 信号 -> T 开盘)
        if pending is not None:
            for sym in list(holdings.keys()):
                if sym in today.index and _is_tradable(today.loc[sym]):
                    fill_px = float(today.loc[sym, "open"]) * (1 - ec.slippage_bp / 1e4)
                    qty = holdings[sym]
                    proceeds = qty * fill_px
                    fee = max(proceeds * ec.sell_commission_bp / 1e4, ec.min_commission)
                    stamp = proceeds * ec.stamp_tax_sell_bp / 1e4
                    cash += proceeds - fee - stamp
                    del holdings[sym]
            equity = cash + sum(q * last_close.get(s, 0.0) for s, q in holdings.items())
            for sym, w in pending.items():
                if sym not in today.index or not _is_tradable(today.loc[sym]):
                    continue
                fill_px = float(today.loc[sym, "open"]) * (1 + ec.slippage_bp / 1e4)
                qty = int(np.floor(equity * w / fill_px / ec.lot_size) * ec.lot_size)
                if qty < ec.lot_size:
                    continue
                cost = qty * fill_px
                fee = max(cost * ec.buy_commission_bp / 1e4, ec.min_commission)
                if cost + fee > cash:
                    continue
                cash -= cost + fee
                holdings[sym] = qty
            pending = None

        for sym in holdings:
            if sym in today.index:
                last_close[sym] = float(today.loc[sym, "close"])

        month = dt.year * 12 + dt.month
        if month != last_month:
            last_month = month
            if "trend_ok" in today.columns:
                ok = list(today.index[today["trend_ok"].fillna(False)])
            else:
                ok = []
            pending = {s: 1.0 / len(ok) for s in ok} if ok else {}

        mv = sum(q * last_close.get(s, 0.0) for s, q in holdings.items())
        equity = cash + mv
        rows.append({"trade_date": dt, "equity": equity, "cash": cash,
                     "market_value": mv, "n_positions": len(holdings),
                     "exposure": (mv / equity) if equity > 0 else 0.0})

    equity_df = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    if bench_df is not None and not equity_df.empty:
        equity_df = _merge_benchmark(equity_df, bench_df, ec.init_cash)
    return {"equity_df": equity_df, "fills_df": pd.DataFrame(),
            "trades_df": pd.DataFrame(), "positions_df": pd.DataFrame(), "closed_trades": []}
