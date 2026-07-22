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

import pandas as pd
import streamlit as st

from app.plots import plot_control_groups, plot_drawdown, plot_equity, plot_symbol_review
from backtest.executor import run_executor
from backtest.metrics import compute_metrics, format_metrics
from backtest.validation import run_control_groups, run_validation
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
    bars = pd.read_parquet(bars_parquet, engine='pyarrow')
    bench = pd.read_parquet(bench_parquet, engine='pyarrow')
    return compute_daily_features(bars, bench, cfg)


def sidebar_params():
    st.sidebar.title("MACD 预判买点 研究台")

    profile = st.sidebar.selectbox(
        "参数档位",
        ["conservative", "neutral", "aggressive"],
        index=1,
        help=(
            "**一键切换风险偏好档位,加载 config/profiles.yaml 中的预设。**\n\n"
            "- **conservative(保守)**:回调前 hist>0 需 5 根,收敛需连续 3 根,stop 线近,信号少胜率高。\n"
            "- **neutral(中性)**:与 default.yaml 完全一致。**默认推荐。**\n"
            "- **aggressive(激进)**:参数放宽,回调长度上限大,只需要连续 1 根收敛,stop 线远。\n\n"
            "切换后下方参数会覆盖档位默认值,方便微调。"
        ),
    )
    cfg = load_config(profile=profile)

    st.sidebar.markdown("#### 信号参数")
    cfg.features.trend_score_min = st.sidebar.slider(
        "趋势分阈值", 3, 6, cfg.features.trend_score_min,
        help=(
            "**六维趋势打分的最小入围值**(每维 1 分,满分 6 分)。\n\n"
            "六个维度:① 收盘 > MA20 ② MA20 > MA60 ③ MA60 > MA120 ④ MA20 斜率>0 "
            "⑤ MA60 斜率>0 ⑥ 20 日收益 > 基准 20 日收益。\n\n"
            "- 3:极宽松,信号最多但假突破多。\n"
            "- 4(默认):中庸,兼顾数量与质量。\n"
            "- 5:只在明确多头排列时才进,交易数骤减。\n"
            "- 6:最严格。"
        ),
    )
    cfg.signal.prior_pos_bars = st.sidebar.slider(
        "回调前 hist>0 最少根数", 1, 10, cfg.signal.prior_pos_bars,
        help=(
            "**回调前需要至少多少根连续的 hist>0(多头动量确立)。**\n\n"
            "- 1~2:抓所有回调,信号最多。\n"
            "- 3(默认):需要至少 3 根绿柱,过滤短促反弹。\n"
            "- 5~10:只抓明显多头趋势中的小回调。"
        ),
    )
    cfg.signal.converge_bars = st.sidebar.slider(
        "hist 收敛连续根数", 1, 5, cfg.signal.converge_bars,
        help=(
            "**hist<0 期间,|hist| 严格递减(越来越接近 0)至少需要多少根才算信号。**\n\n"
            "- 1:任何收窄都算,信号最多。\n"
            "- 2(默认):两连收敛,过滤一日跳空噪声。\n"
            "- 3~5:明确的底背离结构,信号最可靠但最稀。"
        ),
    )
    cfg.signal.max_neg_bars = st.sidebar.slider(
        "回调最长天数", 3, 20, cfg.signal.max_neg_bars,
        help=(
            "**hist<0 连续段超过多少天后,即使收敛也不买入(视为深回调而非短暂回调)。**\n\n"
            "- 5~8(默认):只抓小回调,买在趋势恢复前。\n"
            "- 10~20:深回调也容忍,但可能买在下跌趋势的中期反弹。"
        ),
    )
    cfg.signal.cooldown_days = st.sidebar.slider(
        "清仓后冷却天数", 1, 10, cfg.signal.cooldown_days,
        help="**同一股票清仓后多少天内不再重新买入。**防止同一标的反复来回挨打。",
    )

    st.sidebar.markdown("#### 状态机参数")
    cfg.risk.stop_s1_pct = st.sidebar.slider(
        "S1 止损跌幅(相对 A)", 0.03, 0.12, cfg.risk.stop_s1_pct, 0.005,
        help="**首仓 5% 后,close < A × (1-止损跌幅) 清仓。** 默认 0.06(即 6% 止损)。",
    )
    cfg.risk.add_trigger_pct = st.sidebar.slider(
        "加仓触发涨幅(相对 A)", 0.05, 0.20, cfg.risk.add_trigger_pct, 0.01,
        help="**首仓后,close ≥ A × (1+加仓涨幅) 加 5% 档。** 默认 0.10(10%)。",
    )
    cfg.risk.stop_s2_pct = st.sidebar.slider(
        "S2 止损线(相对 A)", 0.01, 0.08, cfg.risk.stop_s2_pct, 0.005,
        help="**加仓后,close < A × (1+止损线) 全清。** 默认 0.03(回落到 A + 3% 止损)。",
    )
    cfg.risk.tp1_pct = st.sidebar.slider(
        "TP1 减仓线(相对 A)", 0.10, 0.40, cfg.risk.tp1_pct, 0.01,
        help="**close ≥ A × (1+tp1_pct) 减 1/3 仓位。** 默认 0.20(A + 20%)。",
    )
    cfg.risk.lock_inband_pct = st.sidebar.slider(
        "S2 保本回落线", 0.00, 0.05, cfg.risk.lock_inband_pct, 0.005,
        help="**S2 未破 TP1 且 close ≤ A × (1+保本线) 全清。** 默认 0.01(只赚 1% 就走)。",
    )
    cfg.risk.timeout_days = st.sidebar.slider(
        "S1/S2 超时天数", 3, 20, cfg.risk.timeout_days,
        help="**建仓/加仓后,分别在 N 个交易日内未触达加仓/减仓目标 → 全清。** 防止横盘耗损。",
    )
    cfg.risk.tp2_pct = st.sidebar.slider(
        "TP2 减仓线", 0.20, 0.60, cfg.risk.tp2_pct, 0.01,
        help="**减仓后进入 S3,再涨到该线再减 1/3。**",
    )
    cfg.risk.trail_s3_pct = st.sidebar.slider(
        "S3 回落全清线", 0.03, 0.15, cfg.risk.trail_s3_pct, 0.005,
        help="**S3 未破 TP2 且回落到 ≤ A × (1+线值) → 剩余全清。**",
    )
    cfg.risk.tp3_pct = st.sidebar.slider(
        "TP3 全清线", 0.30, 1.00, cfg.risk.tp3_pct, 0.05,
        help="**S4 涨过该线全清。**",
    )
    cfg.risk.trail_s4_pct = st.sidebar.slider(
        "S4 回落全清线", 0.08, 0.30, cfg.risk.trail_s4_pct, 0.005,
        help="**S4 未破 TP3 且回落到 ≤ A × (1+线值) → 剩余全清。**",
    )

    st.sidebar.markdown("#### 仓位")
    cfg.position.fixed_weight = st.sidebar.slider(
        "首仓/每档加仓权重", 0.02, 0.15, cfg.position.fixed_weight, 0.005,
        help="**每档 5% 意味着首仓 5%、加仓再 5%,合计 10%。** 总仓位同时受 max_single_weight 限制。",
    )
    cfg.position.max_single_weight = st.sidebar.slider(
        "单股最大权重", 0.05, 0.25, cfg.position.max_single_weight, 0.01,
        help="**单一股票合计仓位上限。** 默认 0.10(10%),即首仓 + 一档加仓后不再加。",
    )
    cfg.position.max_total_exposure = st.sidebar.slider(
        "总仓位上限", 0.40, 1.00, cfg.position.max_total_exposure, 0.05,
        help="**全部股票总市值/总权益 上限。** 默认 0.80(80%)。",
    )

    st.sidebar.markdown("#### 数据")
    cfg.data.universe_size = st.sidebar.number_input(
        "股票池数量", 10, 300, cfg.data.universe_size,
        help="**从沪深 300(或自定义列表)取前 N 只作为回测标的。** 0 表示全部成分股,需较长下载。",
    )
    cfg.project.start_date = st.sidebar.text_input("开始日期", cfg.project.start_date)
    cfg.project.end_date = st.sidebar.text_input("结束日期", cfg.project.end_date)
    force_refresh = st.sidebar.checkbox("强制重新抓取数据", False)
    run_compare = st.sidebar.checkbox("跑完整对照组 B0-B4", True,
                                      help="勾选后回测时间 ×5,但能看各模块边际贡献。")
    return cfg, force_refresh, run_compare


def metric_cards(m: dict):
    if not m or "error" in m:
        st.warning("无指标")
        return
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    pct = lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) and pd.notna(x) else "n/a"
    c1.metric("年化", pct(m.get("cagr")))
    c2.metric("最大回撤", pct(m.get("max_drawdown")))
    c3.metric("卡玛", f"{m.get('calmar', 0):.2f}" if pd.notna(m.get("calmar")) else "n/a")
    c4.metric("交易胜率", pct(m.get("trade_win_rate")))
    c5.metric("盈亏比", f"{m.get('payoff_ratio', 0):.2f}" if pd.notna(m.get("payoff_ratio")) else "n/a")
    c6.metric("交易数", int(m.get("n_trades", 0)))


def main():
    st.set_page_config(page_title="MACD 预判买点 研究台", layout="wide")
    cfg, force_refresh, run_compare = sidebar_params()

    if st.sidebar.button("运行回测", type="primary"):
        with st.spinner("加载数据..."):
            bars, bench, cal, basic, source = _load(cfg.model_dump(), force_refresh)
        st.session_state["bars"] = bars
        st.session_state["bench"] = bench
        st.session_state["source"] = source
        with st.spinner("计算特征与信号..."):
            # 先把 DataFrame 序列化到临时 Parquet,供缓存去重
            import tempfile
            td = Path(tempfile.mkdtemp())
            (td / "bars.parquet").write_bytes(bars.to_parquet(index=False, engine='pyarrow'))
            (td / "bench.parquet").write_bytes(bench.to_parquet(index=False, engine='pyarrow'))
            feature_df = _features(str(td / "bars.parquet"), str(td / "bench.parquet"), cfg.model_dump())
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
        st.subheader("B4(完整策略)指标")
        metric_cards(metrics["B4"])
        st.code(format_metrics(metrics["B4"], "B4"))
    if run_compare and len(metrics) > 1:
        st.subheader("对照组指标对比")
        rows = []
        for k, mm in metrics.items():
            rows.append({"组": k, "年化": mm.get("cagr"), "最大回撤": mm.get("max_drawdown"),
                         "卡玛": mm.get("calmar"), "夏普": mm.get("sharpe"),
                         "胜率": mm.get("trade_win_rate"), "盈亏比": mm.get("payoff_ratio"),
                         "交易数": mm.get("n_trades"), "超额年化": mm.get("excess_cagr")})
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
