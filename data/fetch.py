"""数据抓取层。

- AKShareProvider(默认):HS300 成分股 + 个股 hfq 日线 + 指数基准 + 交易日历,线程并发抓取 + Parquet 缓存。
- TushareProvider(预留接口):daily + adj_factor + stock_basic + trade_cal + stock_st,需 TUSHARE_TOKEN。
- load_data():统一入口,AKShare 失败时回退合成数据(仅冒烟测试用)。

注意:成分股为当前快照,存在幸存者偏差(已在 README 注明)。
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config_loader import AppConfig
from data import cache as cache_mod
from data.normalize import normalize_index_daily, normalize_stock_daily


def _yyyymmdd(s: str) -> str:
    return s.replace("-", "")[:8]


def _try_import_akshare():
    import akshare as ak  # 延迟导入,避免合成兜底路径强依赖
    return ak


def _with_retry(fn, attempts: int = 4, base_backoff: float = 4.0, label: str = ""):
    """AKShare 源(东财)偶发断连,外层重试 + 递增退避。"""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < attempts - 1:
                wait = base_backoff * (i + 1)
                print(f"  [{label}] 第{i+1}/{attempts} 次失败({repr(e)[:60]}),{wait:.0f}s 后重试")
                time.sleep(wait)
    raise last


# --------------------------------------------------------------------------- #
# AKShare 原始抓取
# --------------------------------------------------------------------------- #
def fetch_hs300_constituents(ak) -> list[str]:
    """获取沪深300成分股代码(当前快照)。"""
    candidates = [
        ("index_stock_cons", ["品种代码", "成分券代码", "代码", "symbol"]),
        ("index_stock_cons_csindex", ["成分券代码", "品种代码", "代码", "symbol"]),
    ]

    def _do():
        last_err = None
        for fn_name, cols in candidates:
            try:
                fn = getattr(ak, fn_name)
                df = fn(symbol="000300")
                for col in cols:
                    if col in df.columns:
                        return df[col].astype(str).str.zfill(6).tolist()
            except Exception as e:  # noqa: BLE001
                last_err = e
        df = ak.hs300_stocks()
        for col in ["code", "symbol", "代码"]:
            if col in df.columns:
                return df[col].astype(str).str.zfill(6).tolist()
        raise RuntimeError(f"无法获取 HS300 成分股: {last_err}")

    return _with_retry(_do, label="hs300_cons")


def fetch_stock_daily(ak, symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    def _do():
        raw = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=_yyyymmdd(start), end_date=_yyyymmdd(end), adjust=adjust,
        )
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        return normalize_stock_daily(raw, symbol)

    try:
        return _with_retry(_do, attempts=4, base_backoff=3.0, label=f"stock {symbol}")
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def fetch_index_daily(ak, symbol: str, start: str, end: str) -> pd.DataFrame:
    def _do():
        raw = ak.index_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=_yyyymmdd(start), end_date=_yyyymmdd(end),
        )
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        return normalize_index_daily(raw, symbol)
    return _with_retry(_do, label=f"index {symbol}")


def fetch_trade_calendar(ak, start: str, end: str) -> pd.DataFrame:
    def _do():
        df = ak.tool_trade_date_hist_sina()
        cal = pd.DataFrame({"trade_date": pd.to_datetime(df["trade_date"])})
        cal = cal[(cal.trade_date >= start) & (cal.trade_date <= end)].sort_values("trade_date")
        cal["prev_trade_date"] = cal["trade_date"].shift(1)
        cal["is_open"] = True
        return cal.reset_index(drop=True)
    return _with_retry(_do, label="calendar")


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #
class AKShareProvider:
    name = "akshare"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def load(self, force_refresh: bool = False):
        cfg = self.cfg
        start, end = cfg.project.start_date, cfg.project.end_date
        s_ymd, e_ymd = _yyyymmdd(start), _yyyymmdd(end)

        cal_name = f"calendar_{s_ymd}_{e_ymd}.parquet"
        bench_name = f"bench_{cfg.project.benchmark}_{cfg.data.adjust}_{s_ymd}_{e_ymd}.parquet"
        bars_name = (f"bars_{cfg.data.symbols_source}_{cfg.data.adjust}"
                     f"_u{cfg.data.universe_size}_{s_ymd}_{e_ymd}.parquet")

        # 1) 交易日历
        if cache_mod.exists(cal_name) and not force_refresh:
            calendar_df = cache_mod.read_parquet(cal_name)
        else:
            ak = _try_import_akshare()
            calendar_df = fetch_trade_calendar(ak, start, end)
            cache_mod.write_parquet(calendar_df, cal_name)

        # 2) 基准指数
        if cache_mod.exists(bench_name) and not force_refresh:
            bench_df = cache_mod.read_parquet(bench_name)
        else:
            ak = _try_import_akshare()
            bench_df = fetch_index_daily(ak, cfg.project.benchmark, start, end)
            cache_mod.write_parquet(bench_df, bench_name)

        # 3) 个股日线
        if cache_mod.exists(bars_name) and not force_refresh:
            bars_df = cache_mod.read_parquet(bars_name)
        else:
            ak = _try_import_akshare()
            symbols = self._universe(ak)
            print(f"[akshare] 抓取 {len(symbols)} 只股票日线 ({start}~{end}, adjust={cfg.data.adjust}) ...")
            bars_df = self._fetch_all_stocks(ak, symbols, start, end, cfg.data.adjust)
            if bars_df.empty:
                raise RuntimeError("AKShare 抓取返回空数据(可能网络受限)。")
            cache_mod.write_parquet(bars_df, bars_name)
            print(f"[akshare] 完成: {bars_df['symbol'].nunique()} 只, {len(bars_df)} 条, 已缓存 -> {bars_name}")

        basic_df = self._basic_df(bars_df)
        return bars_df, bench_df, calendar_df, basic_df

    def _universe(self, ak) -> list[str]:
        cfg = self.cfg
        if cfg.data.symbols_source == "custom" and cfg.data.custom_symbols:
            syms = list(cfg.data.custom_symbols)
        else:
            syms = fetch_hs300_constituents(ak)
        if cfg.data.universe_size and cfg.data.universe_size > 0:
            syms = syms[: cfg.data.universe_size]
        return syms

    def _fetch_all_stocks(self, ak, symbols, start, end, adjust) -> pd.DataFrame:
        out: dict[str, pd.DataFrame] = {}
        done = 0
        # 并发数控制在 3:AKShare 源(东财)对高并发会断连
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(fetch_stock_daily, ak, s, start, end, adjust): s for s in symbols}
            for fut in as_completed(futs):
                sym = futs[fut]
                done += 1
                try:
                    df = fut.result()
                except Exception:  # noqa: BLE001
                    df = pd.DataFrame()
                if len(df) > 0:
                    out[sym] = df
                if done % 10 == 0 or done == len(symbols):
                    print(f"  ... {done}/{len(symbols)} (有效 {len(out)})")
        if not out:
            return pd.DataFrame()
        return (pd.concat(out.values(), ignore_index=True)
                .sort_values(["symbol", "trade_date"]).reset_index(drop=True))

    @staticmethod
    def _basic_df(bars_df: pd.DataFrame) -> pd.DataFrame:
        if bars_df.empty:
            return pd.DataFrame(columns=["symbol", "name", "market", "list_date", "delist_date", "list_status"])
        basic = (bars_df.groupby("symbol")
                 .agg(list_date=("trade_date", "min"), last_date=("trade_date", "max"))
                 .reset_index())
        basic["name"] = basic["symbol"]
        basic["market"] = "A"
        basic["list_status"] = "L"
        basic["delist_date"] = pd.NaT
        return basic


class TushareProvider:
    """预留接口:需 pip install tushare 并设置 TUSHARE_TOKEN。"""
    name = "tushare"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def _pro(self):
        token = os.environ.get("TUSHARE_TOKEN")
        try:
            import tushare as ts
        except ImportError as e:
            raise RuntimeError(
                "Tushare 数据源需要先安装 tushare (pip install tushare) 并设置环境变量 TUSHARE_TOKEN。"
            ) from e
        if not token:
            raise RuntimeError("Tushare 数据源需要设置环境变量 TUSHARE_TOKEN。")
        ts.set_token(token)
        return ts.pro_api()

    def load(self, force_refresh: bool = False):
        cfg = self.cfg
        pro = self._pro()
        start, end = cfg.project.start_date, cfg.project.end_date
        s_ymd, e_ymd = _yyyymmdd(start), _yyyymmdd(end)

        # 交易日历
        cal_raw = pro.trade_cal(exchange="SSE", start_date=s_ymd, end_date=e_ymd)
        cal_raw = cal_raw[cal_raw["is_open"] == 1].sort_values("cal_date")
        calendar_df = pd.DataFrame({
            "trade_date": pd.to_datetime(cal_raw["cal_date"]),
            "is_open": True,
        })
        calendar_df["prev_trade_date"] = calendar_df["trade_date"].shift(1)

        # 基准
        bench_raw = pro.index_daily(ts_code=f"{cfg.project.benchmark}", start_date=s_ymd, end_date=e_ymd)
        bench_df = pd.DataFrame({
            "trade_date": pd.to_datetime(bench_raw["trade_date"]),
            "symbol": cfg.project.benchmark,
            "open": bench_raw["open"].astype(float),
            "high": bench_raw["high"].astype(float),
            "low": bench_raw["low"].astype(float),
            "close": bench_raw["close"].astype(float),
            "volume": bench_raw["vol"].astype(float),
            "amount": bench_raw["amount"].astype(float) * 1000,
        }).sort_values("trade_date")

        # 股票池
        if cfg.data.symbols_source == "custom" and cfg.data.custom_symbols:
            symbols = list(cfg.data.custom_symbols)
        else:
            cons = pro.index_weight(index_code=f"{cfg.project.benchmark}", start_date=s_ymd, end_date=e_ymd)
            symbols = sorted(set(cons["con_symbol"].astype(str)))
            if cfg.data.universe_size and cfg.data.universe_size > 0:
                symbols = symbols[: cfg.data.universe_size]

        frames = []
        for sym in symbols:
            d = pro.daily(ts_code=sym, start_date=s_ymd, end_date=e_ymd)
            if d is None or d.empty:
                continue
            af = pro.adj_factor(ts_code=sym, start_date=s_ymd, end_date=e_ymd)
            d = d.merge(af[["trade_date", "adj_factor"]], on="trade_date", how="left")
            d["adj_factor"] = d["adj_factor"].ffill().fillna(1.0)
            if cfg.data.adjust == "hfq":
                f = d["adj_factor"] / d["adj_factor"].iloc[-1]
            elif cfg.data.adjust == "qfq":
                f = d["adj_factor"] / d["adj_factor"].iloc[0]
            else:
                f = 1.0
            frames.append(pd.DataFrame({
                "trade_date": pd.to_datetime(d["trade_date"]),
                "symbol": sym,
                "open": d["open"].astype(float) * f,
                "high": d["high"].astype(float) * f,
                "low": d["low"].astype(float) * f,
                "close": d["close"].astype(float) * f,
                "volume": d["vol"].astype(float),
                "amount": d["amount"].astype(float) * 1000,
                "adj_factor": d["adj_factor"].astype(float),
            }))
        bars_df = pd.concat(frames, ignore_index=True).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        basic_df = AKShareProvider._basic_df(bars_df)
        return bars_df, bench_df, calendar_df, basic_df


def get_provider(cfg: AppConfig):
    if cfg.data.provider == "tushare":
        return TushareProvider(cfg)
    return AKShareProvider(cfg)


def load_data(cfg: AppConfig, force_refresh: bool = False, allow_synthetic: bool = True):
    """统一入口。返回 (bars_df, bench_df, calendar_df, basic_df, data_source)。"""
    try:
        provider = get_provider(cfg)
        bars_df, bench_df, calendar_df, basic_df = provider.load(force_refresh=force_refresh)
        return bars_df, bench_df, calendar_df, basic_df, provider.name
    except Exception as e:  # noqa: BLE001
        if not allow_synthetic:
            raise
        print(f"[warn] {provider.name} 数据加载失败({e});回退合成数据(仅冒烟测试,结果无研究意义)。")
        return _synthetic_fallback(cfg)


def _synthetic_fallback(cfg: AppConfig):
    import numpy as np
    start = pd.Timestamp(cfg.project.start_date)
    end = pd.Timestamp(cfg.project.end_date)
    calendar_dates = list(pd.bdate_range(start, end))
    calendar_df = pd.DataFrame({
        "trade_date": calendar_dates,
        "is_open": True,
    })
    calendar_df["prev_trade_date"] = calendar_df["trade_date"].shift(1)

    rng = np.random.default_rng(cfg.seed)
    n_syms = min(cfg.data.universe_size or 30, 30)
    symbols = [f"{600000 + i:06d}" for i in range(n_syms)]
    bars_df = cache_mod.generate_synthetic_bars(symbols, calendar_dates, seed=cfg.seed)
    bench_df = cache_mod.generate_synthetic_benchmark(cfg.project.benchmark, calendar_dates, seed=cfg.seed)
    basic_df = AKShareProvider._basic_df(bars_df)
    return bars_df, bench_df, calendar_df, basic_df, "synthetic"
