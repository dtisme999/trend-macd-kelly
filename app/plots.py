"""可视化:Plotly 净值/回撤/个股复盘/对照组对比。"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_equity(equity_df: pd.DataFrame, title: str = "净值曲线") -> go.Figure:
    df = equity_df.sort_values("trade_date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["trade_date"], y=df["equity"], mode="lines", name="策略净值"))
    if "benchmark_equity" in df.columns:
        fig.add_trace(go.Scatter(x=df["trade_date"], y=df["benchmark_equity"], mode="lines", name="基准净值"))
    fig.update_layout(title=title, xaxis_title="日期", yaxis_title="净值",
                      template="plotly_white", height=420)
    return fig


def plot_drawdown(equity_df: pd.DataFrame, title: str = "回撤") -> go.Figure:
    df = equity_df.sort_values("trade_date").copy()
    df["drawdown"] = df["equity"] / df["equity"].cummax() - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["trade_date"], y=df["drawdown"], mode="lines",
                             fill="tozeroy", name="回撤", line=dict(color="crimson")))
    fig.update_layout(title=title, xaxis_title="日期", yaxis_title="回撤",
                      template="plotly_white", height=300)
    return fig


def plot_symbol_review(feature_df: pd.DataFrame, fills_df: pd.DataFrame,
                       symbol: str, title: str = "个股复盘") -> go.Figure:
    df = feature_df[feature_df["symbol"] == symbol].sort_values("trade_date")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.08, subplot_titles=("K线 + MA + 买卖点", "MACD"))
    fig.add_trace(go.Candlestick(x=df["trade_date"], open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="K线"), row=1, col=1)
    for col, color in [("ma_fast", "blue"), ("ma_mid", "orange"), ("ma_slow", "purple")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["trade_date"], y=df[col], mode="lines",
                                     name=col, line=dict(width=1, color=color)), row=1, col=1)
    if fills_df is not None and not fills_df.empty:
        sym_fills = fills_df[fills_df["symbol"] == symbol]
        for action, marker in [("BUY", "triangle-up"), ("SELL", "triangle-down")]:
            sub = sym_fills[sym_fills["action"] == action]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["trade_date"], y=sub["price"], mode="markers",
                                         name=action, marker=dict(symbol=marker, size=11,
                                         color="green" if action == "BUY" else "red")),
                              row=1, col=1)
    if "hist" in df.columns:
        colors = ["green" if h >= 0 else "red" for h in df["hist"]]
        fig.add_trace(go.Bar(x=df["trade_date"], y=df["hist"], name="hist",
                             marker_color=colors, showlegend=False), row=2, col=1)
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, template="plotly_white", height=560)
    return fig


def plot_control_groups(groups: dict, key: str = "equity_df") -> go.Figure:
    fig = go.Figure()
    for name, res in groups.items():
        eq = res.get(key, res) if isinstance(res, dict) else res
        if eq is None or eq.empty:
            continue
        eq = eq.sort_values("trade_date")
        # 统一起点为 1,便于横向比较
        y = eq["equity"] / eq["equity"].iloc[0] if eq["equity"].iloc[0] > 0 else eq["equity"]
        fig.add_trace(go.Scatter(x=eq["trade_date"], y=y, mode="lines", name=name))
    fig.update_layout(title="对照组净值(起点=1)", xaxis_title="日期", yaxis_title="累计净值",
                      template="plotly_white", height=440)
    return fig
