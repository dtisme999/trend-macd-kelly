"""端到端 CLI:抓取 -> 特征 -> 对照组 B0-B4 -> 指标 -> B4 滚动验证 -> 统计检验 -> 存 Parquet -> 打印汇总。

用法:
    python run_backtest.py                         # 中性档,默认股票池
    python run_backtest.py --profile aggressive --universe-size 30
    python run_backtest.py --skip-validation       # 只跑 B4,快速冒烟
    python run_backtest.py --force-refresh         # 忽略缓存重新抓取
"""
from __future__ import annotations

import argparse
import uuid

import pandas as pd

from backtest.executor import run_executor
from backtest.metrics import compute_metrics, format_metrics
from backtest.vector_research import compute_daily_features
from backtest.validation import run_validation
from config_loader import load_config
from data.fetch import load_data
from data.cache import CACHE_DIR
from storage.repo import get_conn, save_run


def _save_run(cfg, name, profile, source, res, metrics):
    try:
        conn = get_conn()
        run_id = f"{name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        save_run(conn, run_id, name, profile, cfg,
                 res["equity_df"], res.get("trades_df", pd.DataFrame()),
                 res.get("fills_df", pd.DataFrame()), metrics, source)
        print(f"  已存库 -> run_id={run_id}")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] 存库失败: {e}")


def _dump(res, tag):
    if not res["equity_df"].empty:
        res["equity_df"].to_parquet(CACHE_DIR / f"equity_{tag}.parquet", index=False)
    tr = res.get("trades_df")
    if tr is not None and not tr.empty:
        tr.to_parquet(CACHE_DIR / f"trades_{tag}.parquet", index=False)
    fl = res.get("fills_df")
    if fl is not None and not fl.empty:
        fl.to_parquet(CACHE_DIR / f"fills_{tag}.parquet", index=False)


def main():
    ap = argparse.ArgumentParser(description="MACD 预判买点 + A 锚点状态机 端到端回测")
    ap.add_argument("--profile", default="neutral", choices=["conservative", "neutral", "aggressive"])
    ap.add_argument("--universe-size", type=int, default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--force-refresh", action="store_true")
    ap.add_argument("--no-synthetic", action="store_true", help="网络失败时不回退合成数据")
    ap.add_argument("--skip-validation", action="store_true", help="只跑 B4,跳过对照组/滚动验证(更快冒烟)")
    args = ap.parse_args()

    cfg = load_config(profile=args.profile)
    if args.universe_size is not None:
        cfg.data.universe_size = args.universe_size
    if args.start:
        cfg.project.start_date = args.start
    if args.end:
        cfg.project.end_date = args.end

    print(f"== MACD 预判买点 状态机策略 | profile={args.profile} | universe={cfg.data.universe_size} ==")
    bars, bench, cal, basic, source = load_data(
        cfg, force_refresh=args.force_refresh, allow_synthetic=not args.no_synthetic)
    print(f"数据来源: {source} | 股票 {bars['symbol'].nunique()} 只 | {len(bars)} 条 | "
          f"{bars['trade_date'].min().date()} ~ {bars['trade_date'].max().date()}")

    print("计算特征与信号...")
    feature_df = compute_daily_features(bars, bench, cfg)
    print(f"  entry_candidate=True 共 {int(feature_df['entry_candidate'].sum())} 个 symbol-日")

    init_cash = cfg.execution.init_cash

    if args.skip_validation:
        res = run_executor(feature_df, cfg, bench_df=bench)
        m = compute_metrics(res["equity_df"], res["trades_df"], res["fills_df"], init_cash)
        print(format_metrics(m, "B4 (skip-validation)"))
        _save_run(cfg, "B4_state_machine", args.profile, source, res, m)
        _dump(res, "B4")
        return

    print("运行对照组 B0-B4 + 滚动验证 + 统计检验...")
    val = run_validation(feature_df, bench, cfg)

    print("\n========== 对照组指标 ==========")
    for k in ["B0", "B1", "B2", "B3", "B4"]:
        print(format_metrics(val["group_metrics"][k], k))
        print()
    _save_run(cfg, "MACD_state_machine_B4", args.profile, source,
              val["groups"]["B4"], val["group_metrics"]["B4"])

    print("========== 统计检验 ==========")
    for k, t in val["t_tests"].items():
        print(f"  配对t {k}: t={t['t']:.3f}  p={t['p']:.4f}  n={t['n']}  mean_diff={t['mean_diff']}")
    if val["dsr"] is not None:
        print(f"  DSR(B4) = {val['dsr']:.4f}   (>0.5 表示经多试验校正后仍显著)")
    if val["spa"]:
        print(f"  SPA: {val['spa']}")
    print("  Bootstrap 95% CI (B4):")
    for name, (lo, hi, pt) in val["bootstrap_ci"].items():
        print(f"    {name:14s}: {pt:.4f}  [{lo:.4f}, {hi:.4f}]")

    print("\n========== 滚动扩窗 B4 ==========")
    wf = val["walk_forward"]
    print(f"  窗口数: {wf['n_windows']}")
    for w in wf["windows"]:
        print(f"    {w['test_start']} ~ {w['test_end']}   训练闭合 {w['train_closed']}   验证交易 {w['test_trades']}")
    if not wf["equity_df"].empty:
        wf_m = compute_metrics(wf["equity_df"], None, None, init_cash)
        print(format_metrics(wf_m, "Walk-forward B4"))
        wf["equity_df"].to_parquet(CACHE_DIR / "equity_walkforward.parquet", index=False)

    for k in ["B0", "B1", "B2", "B3", "B4"]:
        eq = val["groups"][k]["equity_df"]
        if not eq.empty:
            eq.to_parquet(CACHE_DIR / f"equity_{k}.parquet", index=False)
            _dump(val["groups"][k], k)
    print(f"\n净值/交易/成交已导出 -> {CACHE_DIR}/equity_*.parquet")
    print("完成。")


if __name__ == "__main__":
    main()
