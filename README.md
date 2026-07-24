# 精确 MACD 收敛筛选 + 三情景 + A 锚点状态机

A 股日线研究框架，严格执行 MACD 柱段筛选、市场/个股三类趋势对比，以及首仓 5%、
加仓至 10% 的 A 锚点加减仓策略。

## 精确策略口径

- 当前 MACD 柱值为负，连续负柱 2–7 根。
- 当前是连续收敛的第 1 根，即 `abs(hist[t]) < abs(hist[t-1])`，但前一日不是收敛。
- 紧邻负柱段之前的连续正柱不少于 7 根。
- 前正柱最大值 / 当前负柱段截至当日最小值绝对值不低于 2。
- 市场和个股都标记为 `up`、`down`、`sideways`，CLI 和 Parquet 输出分组表现。
- `A` 为首仓实际成交价；`1.10A/1.20A/1.30A/1.40A` 分别表示上涨
  10%/20%/30%/40%。收盘触发，下一交易日开盘成交。
- 模拟阶段按整数股成交，不引入整手与零股申报限制；手续费、滑点和 T+1
  成交仍保留。

四层架构(报告推荐):
- **日线数据层** `data/` — AKShare 抓取 HS300 成分股 hfq 日线 + 指数基准 + 交易日历,Parquet 缓存
- **向量化研究层** `strategy/` + `backtest/vector_research.py` — pandas/NumPy 算指标/信号/参数网格
- **轻量事件执行层** `backtest/executor.py` — 按交易日循环,T 日收盘信号 → T+1 开盘成交,处理费用/仓位/加减仓
- **Python 原生界面层** `app/` + `storage/` — Streamlit + Plotly + DuckDB

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 默认数据源 **AKShare**(免费、无需 token)。若要改用 Tushare:`pip install tushare` 并 `export TUSHARE_TOKEN=...`,再把 `config/default.yaml` 的 `data.provider` 改为 `tushare`。

## 运行

```bash
# 端到端:抓数据 → 对照组 B0-B4 → 指标 → B4 滚动验证 → 统计检验 → 存库
python run_backtest.py --profile neutral --universe-size 50

# 快速冒烟(只跑 B4)
python run_backtest.py --skip-validation --universe-size 20

# 三档对比
python run_backtest.py --profile conservative
python run_backtest.py --profile aggressive

# Streamlit 研究台
streamlit run app/streamlit_app.py
```

输出：`equity_B*.parquet`、`trades_*.parquet`、`fills_*.parquet`、
`market_regime_*.parquet`、`stock_trend_*.parquet` 和 `quant.duckdb`。

## 防校验(硬性,落实为单元测试)

测试覆盖：

1. 指标/信号在 symbol 内 `shift` 界定,扰动未来 K 线不改变任何历史行
2. 所有订单由 T 日信号 → T+1 日开盘价产生(init 成交必有 T-1 信号)
3. MACD 五项筛选的全部边界及强弱比
4. 市场/个股三分类和分组表现可重算
5. S1–S4 每个阈值、5 日边界、固定基数减仓和完整成交路径

```bash
python -m pytest tests/ -v
```

## 目录结构

```
quant/
  config/{default.yaml, profiles.yaml}   # 中性默认 + 保守/中性/激进三档
  config_loader.py                        # pydantic v2 配置 + profile 深合并
  data/{fetch.py, normalize.py, cache.py} # AKShare/Tushare + 归一化 + Parquet/合成兜底
  strategy/{indicators,trend,macd_convergence,rules}.py
  backtest/{vector_research,executor,regime,metrics,validation}.py
  app/{streamlit_app.py, plots.py}        # Streamlit + Plotly
  storage/{schema.sql, repo.py}           # DuckDB
  tests/{test_anti_bias,test_macd_screening,test_position_state_machine,test_regime}.py
  run_backtest.py                          # 端到端 CLI
```

## 固定口径(从第一天写死)

- MACD:`DIF=EMA12-EMA26`,`DEA=EMA9(DIF)`,`hist=2*(DIF-DEA)`
- ATR:Wilder 平滑(RMA,等价 TA-Lib)
- 复权:统一 `hfq`,所有指标/回测只用这一套
- 成交:T 日收盘信号 → T+1 日开盘价,佣金双边 2.5bp + 最低 5 元 + 卖方印花税 10bp；模拟按整数股

## 对照组与统计检验

| 组 | 策略 | 测什么 |
|---|---|---|
| B0 | 基准持有 | 是否跑赢市场 |
| B1 | 趋势筛选等权持有 | 选股层 |
| B2 | 精确 MACD 入场后持有 | 买点层 |
| B3 | 精确 MACD 入场 + 加仓后持有 | 加仓层 |
| B4 | 完整 S1–S4 状态机 | 加减仓与退出层 |

- 配对 t 检验(scipy `ttest_rel`):B4 vs 基准的月度收益差
- DSR(Bailey-López de Prado):纠正多试验选择偏差
- SPA(arch,缺失则跳过):superior predictive ability
- stationary bootstrap(1000 reps):年化/回撤/夏普 95% 置信区间
- 滚动扩窗:训练 3 年 / 验证 1 年 / 步长 1 年

## 已知简化(研究阶段取舍)

- **幸存者偏差**:股票池为 HS300 当前成分快照,未按历史时点调整
- **ST 过滤**:AKShare 无历史 ST,默认关闭;用 Tushare 可开 `data.use_st_filter`(历史 ST 自 2016 起)
- vectorbt 未默认安装(规避 numba/Python 3.13 兼容风险),参数网格用纯 pandas 实现

## 不在本次范围(P2)

每日扫描 `daily_job`(cron)、FastAPI 接口 `api`、浏览器端 Lightweight Charts 复盘。需要时按报告 "最小 API 设计" 增补。
