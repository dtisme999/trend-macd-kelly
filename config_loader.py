"""配置加载:YAML -> pydantic v2 模型,支持 保守/中性/激进 三档 profile 深合并。

口径说明(报告强调"从第一天就固定口径"):
- MACD 柱值 hist = 2 * (DIF - DEA)
- ATR 采用 Wilder 平滑(RMA,等价 TA-Lib)  ← 保留用于 vol20 相关的复盘展示
- 复权口径统一为 hfq
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).resolve().parent / "config"


class ProjectConfig(BaseModel):
    name: str = "trend_macd_state_machine"
    market: str = "cn_stock_daily"
    benchmark: str = "000300"
    start_date: str = "2016-01-01"
    end_date: str = "2026-07-17"


class DataConfig(BaseModel):
    provider: str = "akshare"
    adjust: str = "hfq"
    use_st_filter: bool = False
    min_listing_days: int = 120
    min_avg_amount_20d: float = 50_000_000
    symbols_source: str = "hs300"
    custom_symbols: list[str] = Field(default_factory=list)
    universe_size: int = 50


class FeaturesConfig(BaseModel):
    ma_fast: int = 20
    ma_mid: int = 60
    ma_slow: int = 120
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    vol_period: int = 20
    trend_score_min: int = 4
    regime_fast: int = 20
    regime_slow: int = 60
    regime_slope: int = 20


class SignalConfig(BaseModel):
    """严格 MACD 筛选参数。"""
    prior_pos_bars: int = 7
    min_neg_bars: int = 2
    max_neg_bars: int = 7
    convergence_count: int = 1
    min_strength_ratio: float = 2.0
    cooldown_days: int = 5


class ExecutionConfig(BaseModel):
    init_cash: float = 1_000_000.0
    price_rule: str = "next_open"
    lot_size: int = 1
    slippage_bp: float = 5.0
    buy_commission_bp: float = 2.5
    sell_commission_bp: float = 2.5
    min_commission: float = 5.0
    stamp_tax_sell_bp: float = 10.0


class PositionConfig(BaseModel):
    """极简仓位:首仓与每档加仓都是 fixed_weight。"""
    fixed_weight: float = 0.05
    max_single_weight: float = 0.10       # 首仓 5% + 一档加仓 5%
    max_total_exposure: float = 0.80


class RiskConfig(BaseModel):
    """A 锚点状态机阈值(全部为相对首笔成交价 A 的百分比偏移)。"""
    stop_s1_pct: float = 0.06             # close < 0.94A
    stop_s2_pct: float = 0.03             # close < 1.03A(相对 A 的 +3%)
    add_trigger_pct: float = 0.10         # close ≥ 1.10A → 加仓
    tp1_pct: float = 0.20                 # 1.20A
    tp2_pct: float = 0.30                 # 1.30A
    tp3_pct: float = 0.40                 # 1.40A
    lock_inband_pct: float = 0.01         # S2 未破 1.2A 且回落 1.01A → 保本
    trail_s3_pct: float = 0.08            # S3 未破 1.3A 且回落 1.08A
    trail_s4_pct: float = 0.15            # S4 未破 1.4A 且回落 1.15A
    reduce_fraction_tier: float = 0.3333  # 减仓 1/3(相对总仓 10% => 3.33%)
    timeout_days: int = 5                 # S1/S2 超时
    max_holding_days: int = 120           # 兜底


class ValidationConfig(BaseModel):
    walk_forward: bool = True
    train_years: int = 3
    test_years: int = 1
    step_years: int = 1
    bootstrap_reps: int = 1000
    bootstrap_block_len: int = 20
    bootstrap_type: str = "stationary"
    run_spa_test: bool = True
    run_dsr: bool = True
    n_trials_for_dsr: int = 9


class AppConfig(BaseModel):
    seed: int = 42
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    position: PositionConfig = Field(default_factory=PositionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, profile: str = "neutral") -> AppConfig:
    """加载 default.yaml 并叠加 profile 覆盖项,返回校验后的 AppConfig。"""
    default_path = Path(path) if path else (CONFIG_DIR / "default.yaml")
    with open(default_path, encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}

    profiles_path = CONFIG_DIR / "profiles.yaml"
    if Path(profiles_path).exists():
        with open(profiles_path, encoding="utf-8") as f:
            profiles_doc = yaml.safe_load(f) or {}
        prof = (profiles_doc.get("profiles") or {}).get(profile, {})
        base = _deep_merge(base, prof)

    return AppConfig(**base)


if __name__ == "__main__":
    for p in ("conservative", "neutral", "aggressive"):
        cfg = load_config(profile=p)
        print(f"[{p}] trend_min={cfg.features.trend_score_min} "
              f"prior_pos={cfg.signal.prior_pos_bars} converge={cfg.signal.convergence_count} "
              f"stop_s1={cfg.risk.stop_s1_pct} trail_s3={cfg.risk.trail_s3_pct}")
