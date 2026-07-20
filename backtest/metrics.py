"""指标报告:净值/回撤/交易明细/贡献归因。

输出报告 "回测输出指标表" 中的固定指标集。重点盯:卡玛比率、交易胜率、盈亏比、加仓贡献。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def monthly_returns(equity_df: pd.DataFrame) -> pd.Series:
    df = equity_df.sort_values("trade_date").copy()
    df["ym"] = df["trade_date"].dt.to_period("M")
    return df.groupby("ym")["equity"].last().pct_change().dropna()


def compute_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame | None = None,
                    fills_df: pd.DataFrame | None = None, init_cash: float | None = None,
                    freq: int = TRADING_DAYS) -> dict:
    if equity_df is None or equity_df.empty:
        return {"error": "empty equity"}
    df = equity_df.sort_values("trade_date").reset_index(drop=True)
    init_cash = init_cash if init_cash else df["equity"].iloc[0]
    eq = df["equity"].astype(float)
    n = len(df)
    rets = eq.pct_change().dropna()
    final = float(eq.iloc[-1])
    years = n / freq if freq > 0 else np.nan

    cum = final / init_cash - 1.0
    cagr = (final / init_cash) ** (1.0 / years) - 1.0 if years and years > 0 else 0.0
    cummax = eq.cummax()
    dd = eq / cummax - 1.0
    maxdd = float(dd.min())
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    sharpe = float(rets.mean() / rets.std() * np.sqrt(freq)) if rets.std() > 0 else 0.0

    mret = monthly_returns(df)
    month_win_rate = float((mret > 0).mean()) if len(mret) > 0 else 0.0

    bench = {}
    if "benchmark_equity" in df.columns:
        be = df["benchmark_equity"].astype(float)
        bc = be / be.iloc[0]
        brets = be.pct_change().dropna()
        bench = {
            "bench_cum_return": float(bc.iloc[-1] - 1.0),
            "bench_cagr": float(bc.iloc[-1] ** (1.0 / years) - 1.0) if years and years > 0 else 0.0,
            "bench_max_drawdown": float((bc / bc.cummax() - 1.0).min()),
            "bench_sharpe": float(brets.mean() / brets.std() * np.sqrt(freq)) if brets.std() > 0 else 0.0,
        }
        bench["excess_cagr"] = cagr - bench["bench_cagr"]

    twr = payoff = profit_factor = avg_hold = np.nan
    init_pnl = add_pnl = reduce_prot = 0.0
    n_trades = 0
    if trades_df is not None and len(trades_df) > 0:
        t = trades_df
        n_trades = len(t)
        wins = t[t["return"] > 0]
        losses = t[t["return"] < 0]
        twr = len(wins) / n_trades if n_trades > 0 else 0.0
        if len(wins) > 0 and len(losses) > 0 and losses["return"].mean() != 0:
            payoff = float(wins["return"].mean() / abs(losses["return"].mean()))
        if len(losses) > 0 and losses["realized_pnl"].sum() != 0:
            profit_factor = float(wins["realized_pnl"].sum() / abs(losses["realized_pnl"].sum()))
        avg_hold = float(t["holding_days"].mean())
        init_pnl = float(t["init_pnl"].sum())
        add_pnl = float(t["add_pnl"].sum())
        reduce_prot = float(t["reduce_protection"].sum())

    turnover = np.nan
    if fills_df is not None and len(fills_df) > 0 and years and years > 0:
        notional = float((fills_df["qty"] * fills_df["price"]).sum())
        turnover = notional / (init_cash * years)

    avg_exposure = float(df["exposure"].mean()) if "exposure" in df.columns else np.nan

    return {
        "cum_return": cum, "cagr": cagr, "max_drawdown": maxdd, "calmar": calmar,
        "sharpe": sharpe, "month_win_rate": month_win_rate,
        "trade_win_rate": twr, "payoff_ratio": payoff, "profit_factor": profit_factor,
        "avg_holding_days": avg_hold, "avg_exposure": avg_exposure, "turnover": turnover,
        "n_trades": n_trades,
        "init_contribution": init_pnl, "add_contribution": add_pnl,
        "reduce_protection": reduce_prot,
        "final_equity": final, "n_days": n,
        **bench,
    }


KEY_METRICS = [
    "cagr", "max_drawdown", "calmar", "sharpe", "trade_win_rate",
    "payoff_ratio", "profit_factor", "n_trades", "excess_cagr",
]


def format_metrics(m: dict, title: str = "") -> str:
    if not m or "error" in m:
        return f"{title}: <empty>"
    pct = lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) and np.isfinite(x) else "n/a"
    f4 = lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and np.isfinite(x) else "n/a"
    lines = [f"== {title} =="] if title else []
    lines += [
        f"  累计收益      : {pct(m.get('cum_return'))}",
        f"  年化(CAGR)    : {pct(m.get('cagr'))}    基准: {pct(m.get('bench_cagr'))}    超额: {pct(m.get('excess_cagr'))}",
        f"  最大回撤      : {pct(m.get('max_drawdown'))}    基准: {pct(m.get('bench_max_drawdown'))}",
        f"  卡玛比率      : {f4(m.get('calmar'))}    夏普: {f4(m.get('sharpe'))}    基准夏普: {f4(m.get('bench_sharpe'))}",
        f"  月胜率        : {pct(m.get('month_win_rate'))}",
        f"  交易胜率      : {pct(m.get('trade_win_rate'))}    盈亏比: {f4(m.get('payoff_ratio'))}    盈利因子: {f4(m.get('profit_factor'))}",
        f"  交易数        : {m.get('n_trades')}    平均持仓(天): {f4(m.get('avg_holding_days'))}    平均仓位: {pct(m.get('avg_exposure'))}",
        f"  换手率(年)    : {f4(m.get('turnover'))}",
        f"  首仓贡献      : {f4(m.get('init_contribution'))}    加仓贡献: {f4(m.get('add_contribution'))}    减仓保护: {f4(m.get('reduce_protection'))}",
        f"  终值          : {m.get('final_equity'):.0f}    天数: {m.get('n_days')}",
    ]
    return "\n".join(lines)
