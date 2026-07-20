# 趋势-MACD收敛-动态凯利仓位 · 个人研究版

A 股日线研究框架,验证四件事:**趋势筛选是否给信号提纯、MACD 收敛是否改善入场时机、分批加仓是否改善收益回撤比、简化凯利是否优于固定仓位**。

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

输出:`data/cache/equity_B*.parquet`(净值)、`trades_*.parquet`、`fills_*.parquet`、`quant.duckdb`(回测版本/交易/成交可复现)。

## 防校验(硬性,落实为单元测试)

`tests/test_anti_bias.py`:
1. 指标/信号在 symbol 内 `shift` 界定,扰动未来 K 线不改变任何历史行
2. 所有订单由 T 日信号 → T+1 日开盘价产生(init 成交必有 T-1 信号)
3. 凯利 p̂/b̂ 仅来自已闭合交易,样本不足退回固定仓位

```bash
python -m pytest tests/ -v
```

## 目录结构

```
quant/
  config/{default.yaml, profiles.yaml}   # 中性默认 + 保守/中性/激进三档
  config_loader.py                        # pydantic v2 配置 + profile 深合并
  data/{fetch.py, normalize.py, cache.py} # AKShare/Tushare + 归一化 + Parquet/合成兜底
  strategy/{indicators,trend,macd_convergence,kelly,rules}.py
  backtest/{vector_research,executor,metrics,validation}.py
  app/{streamlit_app.py, plots.py}        # Streamlit + Plotly
  storage/{schema.sql, repo.py}           # DuckDB
  tests/test_anti_bias.py
  run_backtest.py                          # 端到端 CLI
```

## 固定口径(从第一天写死)

- MACD:`DIF=EMA12-EMA26`,`DEA=EMA9(DIF)`,`hist=2*(DIF-DEA)`
- ATR:Wilder 平滑(RMA,等价 TA-Lib)
- 复权:统一 `hfq`,所有指标/回测/凯利样本只用这一套
- 成交:T 日收盘信号 → T+1 日开盘价,佣金双边 2.5bp + 最低 5 元 + 卖方印花税 10bp,100 股取整

## 对照组与统计检验

| 组 | 策略 | 测什么 |
|---|---|---|
| B0 | 基准持有 | 是否跑赢市场 |
| B1 | 趋势筛选等权持有 | 选股层 |
| B2 | +MACD收敛(固定仓位,无加减仓) | 买点层 |
| B3 | +加仓减仓 | 仓位路径 |
| B4 | +动态凯利 | 凯利层 |

- 配对 t 检验(scipy `ttest_rel`):B4 vs 基准、B3 vs B2、B4 vs B3 的月度超额差
- DSR(Bailey-López de Prado):纠正多试验选择偏差
- SPA(arch,缺失则跳过):superior predictive ability
- stationary bootstrap(1000 reps):年化/回撤/夏普 95% 置信区间
- 滚动扩窗:训练 3 年 / 验证 1 年 / 步长 1 年,训练窗估凯利、验证窗冻结

## 已知简化(研究阶段取舍)

- **幸存者偏差**:股票池为 HS300 当前成分快照,未按历史时点调整
- **ST 过滤**:AKShare 无历史 ST,默认关闭;用 Tushare 可开 `data.use_st_filter`(历史 ST 自 2016 起)
- **凯利在线估计**:B4 全周期用在线凯利(随交易更新);滚动验证窗用冻结凯利
- vectorbt 未默认安装(规避 numba/Python 3.13 兼容风险),参数网格用纯 pandas 实现

## 不在本次范围(P2)

每日扫描 `daily_job`(cron)、FastAPI 接口 `api`、浏览器端 Lightweight Charts 复盘。需要时按报告 "最小 API 设计" 增补。
