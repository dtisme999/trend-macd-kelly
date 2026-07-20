-- DuckDB schema(文件即数据库,落 data/cache/quant.duckdb)
-- 表结构对齐报告 "示例数据表结构",并按实际回测产出字段扩展。

CREATE TABLE IF NOT EXISTS backtest_runs (
  run_id        TEXT PRIMARY KEY,
  strategy_name TEXT,
  profile       TEXT,
  config_json   TEXT,
  metrics_json  TEXT,
  created_at    TIMESTAMP,
  start_date    DATE,
  end_date      DATE,
  benchmark     TEXT,
  data_source   TEXT
);

CREATE TABLE IF NOT EXISTS equity (
  run_id           TEXT,
  trade_date       DATE,
  equity           DOUBLE,
  cash             DOUBLE,
  market_value     DOUBLE,
  n_positions      INTEGER,
  exposure         DOUBLE,
  benchmark_equity DOUBLE,
  PRIMARY KEY (run_id, trade_date)
);

CREATE TABLE IF NOT EXISTS trades (
  run_id            TEXT,
  symbol            TEXT,
  entry_date        DATE,
  exit_date         DATE,
  holding_days      INTEGER,
  invested_cost     DOUBLE,
  realized_pnl      DOUBLE,
  return            DOUBLE,
  entry_trend_score INTEGER,
  entry_vol20       DOUBLE,
  init_pnl          DOUBLE,
  add_pnl           DOUBLE,
  reduce_protection DOUBLE,
  n_adds            INTEGER,
  weight            DOUBLE
);

CREATE TABLE IF NOT EXISTS fills (
  run_id      TEXT,
  trade_date  DATE,
  symbol      TEXT,
  action      TEXT,
  order_type  TEXT,
  lot_type    TEXT,
  price       DOUBLE,
  qty         INTEGER,
  fee         DOUBLE,
  stamp       DOUBLE,
  reason      TEXT
);

CREATE TABLE IF NOT EXISTS features_daily (
  trade_date   DATE,
  symbol       TEXT,
  ma20         DOUBLE,
  ma60         DOUBLE,
  ma120        DOUBLE,
  dif          DOUBLE,
  dea          DOUBLE,
  hist         DOUBLE,
  atr14        DOUBLE,
  vol20        DOUBLE,
  trend_score  INTEGER,
  conv_n       INTEGER,
  PRIMARY KEY (trade_date, symbol)
);
