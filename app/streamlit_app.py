"""Streamlit 单页研究台:配参数 -> 跑回测 -> 看复盘。

运行:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# streamlit run 只把脚本所在目录(app/)加入 sys.path,而非项目根目录,
# 导致 `from app.plots import ...` 这类绝对导入失败。把项目根补进 sys.path。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy

import pandas as pd
import streamlit as st

from app.plots import plot_control_groups, plot_drawdown, plot_equity, plot_symbol_review
from backtest.executor import run_executor
from backtest.metrics import compute_metrics, format_metrics
from backtest.validation import run_control_groups
from backtest.vector_research import compute_daily_features
from config_loader import load_config
from data.fetch import load_data


@st.cache_data(show_spinner=True)
def _load(cfg_dict: dict, force_refresh: bool):
    from config_loader import AppConfig
    cfg = AppConfig(**cfg_dict)
    bars, bench, cal, basic, source = load_data(cfg, force_refresh=force_refresh, allow_synthetic=True)
    return bars, bench, cal, basic, source


@st.cache_data(show_spinner=True)
def _features(bars_parquet, bench_parquet, cfg_dict):
    from config_loader import AppConfig
    cfg = AppConfig(**cfg_dict)
    bars = pd.read_parquet(bars_parquet)
    bench = pd.read_parquet(bench_parquet)
    return compute_daily_features(bars, bench, cfg)


def sidebar_params():
    st.sidebar.title("趋势-MACD-凯利 研究台")
    profile = st.sidebar.selectbox("参数档位", ["conservative", "neutral", "aggressive"], index=1)
    cfg = load_config(profile=profile)

    cfg.features.trend_score_min = st.sidebar.slider("趋势分阈值", 3, 6, cfg.features.trend_score_min)
    cfg.signal.wait_bars_after_death_cross = st.sidebar.slider("死叉后等待天数", 1, 5, cfg.signal.wait_bars_after_death_cross)
    cfg.signal.convergence_bars = st.sidebar.slider("连续收敛根数", 1, 3, cfg.signal.convergence_bars)
    cfg.position.mode = st.sidebar.selectbox("仓位模式", ["fixed", "fractional_kelly"],
                                             index=0 if cfg.position.mode == "fixed" else 1)
    cfg.position.kelly_fraction = st.sidebar.slider("凯利折扣", 0.1, 0.5, cfg.position.kelly_fraction, 0.05)
    cfg.data.universe_size = st.sidebar.number_input("股票池数量", 10, 300, cfg.data.universe_size)
    cfg.project.start_date = st.sidebar.text_input("开始日期", cfg.project.start_date)
    cfg.project.end_date = st.sidebar.text_input("结束日期", cfg.project.end_date)
    force_refresh = st.sidebar.checkbox("强制重新抓取数据", False)
    run_compare = st.sidebar.checkbox("跑完整对照组 B0-B4", True)
    return cfg, force_refresh, run_compare


def metric_cards(m: dict):
    if not m or "error" in m:
        st.warning("无指标")
        return
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    pct = lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) and pd.notna(x) else "n/a"
    c1.metric("CAGR", pct(m.get("cagr")))
    c2.metric("最大回撤", pct(m.get("max_drawdown")))
    c3.metric("Calmar", f"{m.get('calmar', 0):.2f}" if pd.notna(m.get("calmar")) else "n/a")
    c4.metric("交易胜率", pct(m.get("trade_win_rate")))
    c5.metric("盈亏比", f"{m.get('payoff_ratio', 0):.2f}" if pd.notna(m.get("payoff_ratio")) else "n/a")
    c6.metric("交易数", int(m.get("n_trades", 0)))


def main():
    st.set_page_config(page_title="趋势-MACD-凯利 研究台", layout="wide")
    cfg, force_refresh, run_compare = sidebar_params()

    if st.sidebar.button("运行回测", type="primary"):
        with st.spinner("加载数据..."):
            bars, bench, cal, basic, source = _load(cfg.model_dump(), force_refresh)
        st.session_state["bars"] = bars
        st.session_state["bench"] = bench
        st.session_state["source"] = source
        with st.spinner("计算特征与信号..."):
            feature_df = compute_daily_features(bars, bench, cfg)
        st.session_state["feature_df"] = feature_df

        if run_compare:
            with st.spinner("运行对照组 B0-B4..."):
                groups, metrics = run_control_groups(feature_df, bench, cfg)
            st.session_state["groups"] = groups
            st.session_state["metrics"] = metrics
            st.session_state["b4"] = groups["B4"]
        else:
            with st.spinner("运行回测..."):
                res = run_executor(feature_df, cfg, bench_df=bench)
            st.session_state["b4"] = res
            st.session_state["metrics"] = {"B4": compute_metrics(res["equity_df"], res["trades_df"], res["fills_df"], cfg.execution.init_cash)}

    if "feature_df" not in st.session_state:
        st.info("在左侧配置参数后点击「运行回测」。")
        return

    st.caption(f"数据来源: {st.session_state.get('source', '?')}")
    metrics = st.session_state.get("metrics", {})
    if "B4" in metrics:
        st.subheader("B4(趋势+MACD+加减仓+凯利)指标")
        metric_cards(metrics["B4"])
        st.code(format_metrics(metrics["B4"], "B4"))
    if run_compare and len(metrics) > 1:
        st.subheader("对照组指标对比")
        import pandas as pd
        rows = []
        for k, mm in metrics.items():
            rows.append({"组": k, "CAGR": mm.get("cagr"), "MaxDD": mm.get("max_drawdown"),
                         "Calmar": mm.get("calmar"), "Sharpe": mm.get("sharpe"),
                         "胜率": mm.get("trade_win_rate"), "盈亏比": mm.get("payoff_ratio"),
                         "交易数": mm.get("n_trades"), "超额CAGR": mm.get("excess_cagr")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    b4 = st.session_state.get("b4", {})
    eq = b4.get("equity_df")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["净值", "回撤", "交易列表", "个股复盘", "对照组"])
    with tab1:
        if eq is not None and not eq.empty:
            st.plotly_chart(plot_equity(eq), use_container_width=True)
    with tab2:
        if eq is not None and not eq.empty:
            st.plotly_chart(plot_drawdown(eq), use_container_width=True)
    with tab3:
        tr = b4.get("trades_df")
        if tr is not None and not tr.empty:
            st.dataframe(tr, use_container_width=True)
        else:
            st.info("无闭合交易")
    with tab4:
        feat = st.session_state["feature_df"]
        syms = sorted(feat["symbol"].unique().tolist())
        sym = st.selectbox("选择股票", syms)
        fills = b4.get("fills_df")
        if sym:
            st.plotly_chart(plot_symbol_review(feat, fills if fills is not None else pd.DataFrame(), sym), use_container_width=True)
    with tab5:
        groups = st.session_state.get("groups")
        if groups:
            st.plotly_chart(plot_control_groups(groups), use_container_width=True)


if __name__ == "__main__":
    main()
