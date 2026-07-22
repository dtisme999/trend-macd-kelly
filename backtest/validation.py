"""验证层:对照组 B0-B4 / 滚动扩窗 / 配对 t 检验 / DSR / SPA / stationary bootstrap。

对照组(测各层是否真有增量):
  B0 基准持有 | B1 趋势筛选等权 | B2 预判买点入场(无加减仓) | B3 + 加减仓 | B4 + 全状态机

统计检验三层:
  1. 配对 t 检验(月度超额收益差,scipy ttest_rel)
  2. SPA(arch,缺失则跳过)
  3. DSR(Bailey-López de Prado,纠正多试验选择偏差)
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import norm, skew, ttest_rel

from backtest.executor import (
    _merge_benchmark,
    run_benchmark_hold,
    run_equal_weight_trend,
    run_executor,
)
from backtest.metrics import compute_metrics, monthly_returns


# --------------------------------------------------------------------------- #
# 对照组
# --------------------------------------------------------------------------- #
def run_control_groups(feature_df, bench_df, cfg, seed=42):
    init_cash = cfg.execution.init_cash
    groups = {}

    groups["B0"] = {
        "equity_df": run_benchmark_hold(bench_df, init_cash),
        "trades_df": pd.DataFrame(), "fills_df": pd.DataFrame(),
    }
    groups["B1"] = run_equal_weight_trend(feature_df, cfg, bench_df=bench_df, seed=seed)

    # B2:只有入场信号,不加仓不减仓不止损(纯看入场质量)
    groups["B2"] = run_executor(feature_df, cfg, use_macd=True, use_exits=False,
                                use_adds=False, use_reduces=False,
                                bench_df=bench_df, seed=seed)
    # B3:入场 + 加仓(不含退出)
    groups["B3"] = run_executor(feature_df, cfg, use_macd=True, use_exits=False,
                                use_adds=True, use_reduces=False,
                                bench_df=bench_df, seed=seed)
    # B4:完整策略(入场 + 加仓 + 减仓 + 止损 + 超时 + 保本)
    groups["B4"] = run_executor(feature_df, cfg, use_macd=True, use_exits=True,
                                use_adds=True, use_reduces=True,
                                bench_df=bench_df, seed=seed)

    metrics = {k: compute_metrics(r["equity_df"], r.get("trades_df"), r.get("fills_df"), init_cash)
               for k, r in groups.items()}
    return groups, metrics


# --------------------------------------------------------------------------- #
# 滚动扩窗
# --------------------------------------------------------------------------- #
def walk_forward(feature_df, bench_df, cfg, seed=42):
    vc = cfg.validation
    dates = sorted(feature_df["trade_date"].unique())
    if not dates:
        return {"equity_df": pd.DataFrame(), "windows": [], "n_windows": 0}
    start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    train_td = pd.Timedelta(days=int(vc.train_years * 365))
    test_td = pd.Timedelta(days=int(vc.test_years * 365))
    step_td = pd.Timedelta(days=int(vc.step_years * 365))

    windows, t0 = [], start
    while True:
        train_end = t0 + train_td
        test_end = train_end + test_td
        if test_end > end:
            break
        windows.append((t0, train_end, test_end))
        t0 = t0 + step_td

    test_equities, stats = [], []
    for (t0, train_end, test_end) in windows:
        train_df = feature_df[(feature_df.trade_date >= t0) & (feature_df.trade_date < train_end)]
        test_df = feature_df[(feature_df.trade_date >= train_end) & (feature_df.trade_date < test_end)]
        if train_df.empty or test_df.empty:
            continue
        # 训练窗跑一遍确认信号逻辑;验证窗直接跑完整策略(不再需要 kelly 估计)
        train_res = run_executor(train_df, cfg, use_macd=True, use_exits=True, use_adds=True,
                                 use_reduces=True, seed=seed)
        test_res = run_executor(test_df, cfg, use_macd=True, use_exits=True, use_adds=True,
                                use_reduces=True, seed=seed)
        if test_res["equity_df"].empty:
            continue
        test_equities.append(test_res["equity_df"])
        stats.append({
            "train_start": str(t0.date()), "train_end": str(train_end.date()),
            "test_start": str(train_end.date()), "test_end": str(test_end.date()),
            "train_closed": int(len(train_res.get("closed_trades", []))),
            "test_trades": int(len(test_res.get("trades_df", []))),
        })

    if not test_equities:
        return {"equity_df": pd.DataFrame(), "windows": stats, "n_windows": 0}
    combined = _chain_equity(test_equities, cfg.execution.init_cash)
    if bench_df is not None and not combined.empty:
        combined = _merge_benchmark(combined, bench_df, cfg.execution.init_cash)
    return {"equity_df": combined, "windows": stats, "n_windows": len(test_equities)}


def _chain_equity(equity_dfs, init_cash):
    out, scale = [], init_cash
    for eq in equity_dfs:
        eq = eq.sort_values("trade_date").reset_index(drop=True)
        first = eq["equity"].iloc[0]
        if first <= 0:
            first = 1.0
        ratio = scale / first
        eq = eq[["trade_date", "equity", "cash", "market_value", "n_positions", "exposure"]].copy()
        eq["equity"] = eq["equity"] * ratio
        out.append(eq)
        scale = float(eq["equity"].iloc[-1])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------------------- #
# 统计检验
# --------------------------------------------------------------------------- #
def paired_ttest(strat_monthly: pd.Series, bench_monthly: pd.Series) -> dict:
    df = pd.concat([strat_monthly.rename("s"), bench_monthly.rename("b")], axis=1).dropna()
    if len(df) < 2:
        return {"t": np.nan, "p": np.nan, "n": int(len(df)), "mean_diff": np.nan}
    diff = df["s"] - df["b"]
    if diff.std() == 0:
        return {"t": 0.0, "p": 1.0, "n": int(len(df)), "mean_diff": float(diff.mean()),
                "note": "无差异(两序列月度收益完全一致)"}
    t, p = ttest_rel(df["s"], df["b"])
    return {"t": float(t), "p": float(p), "n": int(len(df)),
            "mean_diff": float(diff.mean())}


def deflated_sharpe_ratio(sharpe: float, n_trials: int, n_obs: int,
                          skew_val: float, kurt_val: float) -> float:
    """Bailey & López de Prado DSR。

    sharpe 为非年化(日)夏普 mean/std;n_obs 为观测数。
    """
    if n_obs < 2:
        return 0.0
    if n_trials <= 1:
        sr0 = 0.0
    else:
        e = np.sqrt(2 * np.log(n_trials))
        emc = 0.5772156649015329  # Euler-Mascheroni
        e_max = e - (emc - np.log(2 * np.pi) + np.log(np.log(n_trials))) / (2 * e)
        sr0 = e_max / np.sqrt(n_obs)
    num = (sharpe - sr0) * np.sqrt(n_obs - 1)
    den = np.sqrt(1 - skew_val * sharpe + ((kurt_val - 3) / 4) * sharpe ** 2)
    if den <= 0:
        return 0.0
    return float(norm.cdf(num / den))


def spa_test(strategy_monthly: pd.Series, bench_monthly: pd.Series, reps: int = 1000) -> dict:
    try:
        from arch.bootstrap import SPA
    except Exception:  # noqa: BLE001
        return {"p": None, "note": "arch 未安装或导入失败,SPA 跳过"}
    try:
        df = pd.concat([strategy_monthly.rename("s"), bench_monthly.rename("b")], axis=1).dropna()
        if len(df) < 30:
            return {"p": None, "note": "样本不足,SPA 跳过"}
        np.random.seed(42)  # arch 不接受 random_state,统一设种子
        spa = SPA(df["b"].values, df["s"].values.reshape(-1, 1),
                  bootstrap="stationary", reps=reps)
        spa.compute()
        p = float(np.asarray(spa.pvalues).flat[0])  # arch 8.0: pvalues(数组)
        return {"p": p, "note": "SPA (stationary bootstrap)"}
    except Exception as e:  # noqa: BLE001
        return {"p": None, "note": f"SPA 计算失败: {e}"}


# ---- stationary bootstrap(纯 numpy,arch 缺失时兜底) ---- #
def stationary_bootstrap_ci(returns: pd.Series, stat_fn, reps=1000, block_len=20, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(returns.dropna(), dtype=float)
    n = len(x)
    if n < block_len:
        return (np.nan, np.nan, np.nan)
    p = 1.0 / block_len
    stats = np.empty(reps)
    for r in range(reps):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        for i in range(1, n):
            if rng.random() < p:
                idx[i] = rng.integers(n)
            else:
                v = idx[i - 1] + 1
                idx[i] = v if v < n else v - n
        stats[r] = stat_fn(x[idx])
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)),
            float(stat_fn(x)))


def _ann_return(r):
    if len(r) == 0:
        return 0.0
    return float((np.prod(1 + r)) ** (252 / len(r)) - 1)


def _maxdd(r):
    if len(r) == 0:
        return 0.0
    wealth = np.cumprod(1 + r)
    return float((wealth / np.maximum.accumulate(wealth) - 1).min())


def _sharpe(r):
    if len(r) < 2 or np.std(r) == 0:
        return 0.0
    return float(np.mean(r) / np.std(r) * np.sqrt(252))


# --------------------------------------------------------------------------- #
# 总入口
# --------------------------------------------------------------------------- #
def run_validation(feature_df, bench_df, cfg, seed=42):
    vc = cfg.validation
    groups, group_metrics = run_control_groups(feature_df, bench_df, cfg, seed)

    b4_eq = groups["B4"]["equity_df"]
    b0_eq = groups["B0"]["equity_df"]

    b4_m = monthly_returns(b4_eq)
    b0_m = monthly_returns(b0_eq)

    t_tests = {
        "B4_vs_B0": paired_ttest(b4_m, b0_m),
    }

    dsr = None
    if vc.run_dsr and not b4_eq.empty:
        rets = b4_eq["equity"].pct_change().dropna()
        if len(rets) > 30:
            sk = float(skew(rets))
            ku = float(sp_kurtosis(rets, fisher=True))  # excess kurtosis
            daily_sharpe = float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0
            dsr = deflated_sharpe_ratio(daily_sharpe, vc.n_trials_for_dsr, len(rets), sk, ku + 3)

    spa = spa_test(b4_m, b0_m, reps=vc.bootstrap_reps) if vc.run_spa_test else None

    boot = {}
    if not b4_eq.empty:
        rets = b4_eq["equity"].pct_change().dropna()
        if len(rets) > 30:
            for name, fn in [("ann_return", _ann_return), ("max_drawdown", _maxdd), ("sharpe", _sharpe)]:
                boot[name] = stationary_bootstrap_ci(
                    rets, fn, vc.bootstrap_reps, vc.bootstrap_block_len, seed)

    wf = {"n_windows": 0, "equity_df": pd.DataFrame(), "windows": []}
    if vc.walk_forward:
        wf = walk_forward(feature_df, bench_df, cfg, seed)

    return {
        "group_metrics": group_metrics,
        "groups": groups,
        "t_tests": t_tests,
        "dsr": dsr,
        "spa": spa,
        "bootstrap_ci": boot,
        "walk_forward": wf,
    }
