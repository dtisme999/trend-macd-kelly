"""配置加载:YAML -> pydantic v2 模型,支持 保守/中性/激进 三档 profile 深合并。

口径说明(报告强调"从第一天就固定口径"):
- MACD 柱值 hist = 2 * (DIF - DEA)
- ATR 采用 Wilder 平滑(RMA,等价 TA-Lib)
- 复权口径统一为 hfq,所有指标/回测/凯利样本只用这一套
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).resolve().parent / "config"


class ProjectConfig(BaseModel):
    name: str = "trend_macd_kelly_personal"
    market: str = "cn_stock_daily"
    benchmark: str = "000300"
    start_date: str = "2016-01-01"
    end_date: str = "2026-07-17"


class DataConfig(BaseModel):
    provider: str = "akshare"            # akshare | tushare
    adjust: str = "hfq"                  # hfq | qfq | raw
    use_st_filter: bool = False
    min_listing_days: int = 120
    min_avg_amount_20d: float = 50_000_000
    symbols_source: str = "hs300"        # hs300 | custom
    custom_symbols: list[str] = Field(default_factory=list)
    universe_size: int = 50              # 0 = 全部成分股


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


class SignalConfig(BaseModel):
    wait_bars_after_death_cross: int = 2
    convergence_bars: int = 2
    require_hist_below_zero: bool = True
    require_dif_turn_up: bool = False
    cooldown_days: int = 5


class ExecutionConfig(BaseModel):
    init_cash: float = 1_000_000.0
    price_rule: str = "next_open"
    lot_size: int = 100
    slippage_bp: float = 5.0
    buy_commission_bp: float = 2.5
    sell_commission_bp: float = 2.5
    min_commission: float = 5.0
    stamp_tax_sell_bp: float = 10.0


class PositionConfig(BaseModel):
    mode: str = "fractional_kelly"       # fixed | fractional_kelly
    fixed_weight: float = 0.05
    kelly_fraction: float = 0.25
    kelly_min_samples: int = 20
    kelly_weight_min: float = 0.02
    kelly_weight_max: float = 0.10
    fallback_fixed_weight: float = 0.05
    init_ratio: float = 0.40
    add1_target_ratio: float = 0.70
    add2_target_ratio: float = 1.00
    add1_mode: str = "breakout"          # breakout | profit
    add1_profit_pct: float = 0.03
    add2_profit_atr: float = 1.0
    vol_target: float = 0.02             # vol_adj 的 sigma*
    max_single_weight: float = 0.10
    max_total_exposure: float = 0.80


class RiskConfig(BaseModel):
    init_stop_atr: float = 2.0
    drawdown_reduce_1: float = 0.05
    drawdown_reduce_2: float = 0.08
    drawdown_exit: float = 0.12
    reduce_fraction_1: float = 0.33
    reduce_fraction_2: float = 0.50
    max_holding_days: int = 120
    ineffective_holding_days: int = 60
    min_effective_return: float = 0.03


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
    n_trials_for_dsr: int = 27


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
              f"wait={cfg.signal.wait_bars_after_death_cross} "
              f"init_ratio={cfg.position.init_ratio} "
              f"stop_atr={cfg.risk.init_stop_atr} "
              f"max_exp={cfg.position.max_total_exposure}")
