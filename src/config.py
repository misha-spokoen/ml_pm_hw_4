"""Параметры сценариев. Значения совпадают с preregistration.md."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
MODELS = ROOT / "models"

SEED = 20261101
MONTHS = pd.date_range("2021-01-01", periods=66, freq="MS")  # индекс месяца t = 0..65
TRAIN_CUTS = list(range(25, 37))  # февраль 2023 — январь 2024
CAL_CUTS = [41, 42, 43]  # июнь–август 2024
TEST_CUTS = list(range(48, 60))  # январь–декабрь 2025

INDUSTRIES = ["trade", "services", "construction", "tourism", "it"]
INDUSTRY_P = [.30, .30, .15, .10, .15]
AMPLITUDE = [.08, .12, .35, .55, .05]
PEAK = [11, 3, 6, 7, 0]  # месяц года с максимальным оборотом, 0 = январь
HIBERNATE = [0., 0., .10, .25, 0.]
SEASONAL = ("construction", "tourism")
REASONS = ["price_fee", "price_fixed", "credit_counterparty", "service", "other", "forced"]

MARGIN = .5
OPS_PER_COHORT = 150_000
DEV_COST = 3_150_000
COHORT_SELECTED = 1_000
TREATED_SHARE = .9


@dataclass(frozen=True)
class Scenario:
    name: str = "S0"
    n: int = 700_000
    seed: int = SEED
    annual_churn: float = .075
    manageable_share: float = .45
    save_rate: float = .35
    persistence_days: float = 240.
    dog_base: float = .004
    dog_extra: float = .02
    hidden_forced: float = .003
    latent_sigma: float = .8


S0 = Scenario()
SENSITIVITY = [
    replace(S0, name="churn15", annual_churn=.15, n=300_000),
    replace(S0, name="churn69", annual_churn=.69, n=300_000),
    replace(S0, name="save20", save_rate=.20, n=300_000),
    replace(S0, name="save50", save_rate=.50, n=300_000),
]
DEV = replace(S0, name="dev", n=60_000, seed=999)
