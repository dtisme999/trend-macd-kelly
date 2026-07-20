"""DuckDB 仓储:回测版本/配置/净值/交易/成交可复现落库与查询。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from data.cache import CACHE_DIR

DB_PATH = CACHE_DIR / "quant.duckdb"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_conn(db_path: Path | str | None = None):
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def _insert_df(conn, df: pd.DataFrame, table: str):
    if df is None or df.empty:
        return
    # 按表定义的列顺序对齐(INSERT ... SELECT * 按位置匹配)
    table_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    df2 = df.reindex(columns=table_cols)
    conn.register("_tmp_insert", df2)
    conn.execute(f"INSERT INTO {table} SELECT * FROM _tmp_insert")
    conn.unregister("_tmp_insert")


def save_run(conn, run_id: str, strategy_name: str, profile: str, cfg,
             equity_df: pd.DataFrame, trades_df: pd.DataFrame, fills_df: pd.DataFrame,
             metrics: dict, data_source: str = "") -> str:
    cfg_json = cfg.model_dump_json(indent=2) if hasattr(cfg, "model_dump_json") else json.dumps(str(cfg))
    metrics_json = json.dumps(metrics, default=str, ensure_ascii=False)
    conn.execute(
        "INSERT INTO backtest_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
        [run_id, strategy_name, profile, cfg_json, metrics_json, datetime.now(),
         pd.Timestamp(cfg.project.start_date), pd.Timestamp(cfg.project.end_date),
         cfg.project.benchmark, data_source],
    )
    eq = equity_df.copy() if equity_df is not None and not equity_df.empty else pd.DataFrame()
    if not eq.empty:
        eq = eq.assign(run_id=run_id)
        cols = ["run_id", "trade_date", "equity", "cash", "market_value",
                "n_positions", "exposure"] + (["benchmark_equity"] if "benchmark_equity" in eq.columns else [])
        _insert_df(conn, eq[cols], "equity")
    if trades_df is not None and not trades_df.empty:
        _insert_df(conn, trades_df.assign(run_id=run_id), "trades")
    if fills_df is not None and not fills_df.empty:
        _insert_df(conn, fills_df.assign(run_id=run_id), "fills")
    return run_id


def list_runs(conn) -> pd.DataFrame:
    return conn.execute(
        "SELECT run_id, strategy_name, profile, created_at, start_date, end_date, benchmark, data_source "
        "FROM backtest_runs ORDER BY created_at DESC"
    ).fetchdf()


def load_metrics(conn, run_id: str) -> dict:
    row = conn.execute("SELECT metrics_json FROM backtest_runs WHERE run_id=?", [run_id]).fetchone()
    return json.loads(row[0]) if row and row[0] else {}


def load_equity(conn, run_id: str) -> pd.DataFrame:
    return conn.execute("SELECT * FROM equity WHERE run_id=? ORDER BY trade_date", [run_id]).fetchdf()


def load_trades(conn, run_id: str) -> pd.DataFrame:
    return conn.execute("SELECT * FROM trades WHERE run_id=? ORDER BY exit_date", [run_id]).fetchdf()
