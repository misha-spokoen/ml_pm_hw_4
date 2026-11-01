"""Слой решений. Гардрейлы детерминированы и работают независимо от качества модели."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import INDUSTRIES, MARGIN, SEASONAL
from legacy import OFFERS, psi, quote_vec, rule_score
from models import FEATURES_V2

MONITORED = ["season_adjusted_ratio", "yoy_ratio_3m", "fee_burden", "self_transfer_share", "payments", "cash_buffer"]
FLAGS = ["compliance", "liquidation", "tochka", "consent", "days_last_offer"]
EXTRA = ["client_id", "cut", "turnover_ratio", "revenue_year"]
RANGES = {"season_adjusted_ratio": (0, 10), "yoy_ratio": (0, 10), "yoy_ratio_3m": (0, 10), "last_year_forward_ratio": (0, 10),
          "days_no_incoming": (0, 366), "fee_burden": (0, 3), "self_transfer_share": (0, 1), "complaints": (0, 100),
          "services": (0, 20), "tenure": (0, 600), "payments": (0, 5000), "cash_buffer": (0, 1)}
REVENUE_LIMIT = 1_500_000
SEASONAL_DROP = .4
MISSING_LIMIT = .1
PSI_LIMIT = .2
MIN_BATCH = 1000  # на меньшей партии PSI отражает отбор, а не сдвиг данных


class SchemaError(ValueError):
    """Партия не обрабатывается: нет обязательных столбцов."""


def batch_gate(df, reference):
    """Проверки всей партии: сдвиг и пропуски блокируют автоматические офферы."""
    drift = {c: psi(reference[c], df[c].fillna(reference[c].median())) for c in MONITORED}
    missing = float(df[FEATURES_V2].isna().mean().max())
    reasons = [f"PSI {c}={v:.2f}" for c, v in drift.items() if v > PSI_LIMIT] if len(df) >= MIN_BATCH else []
    if missing > MISSING_LIMIT:
        reasons.append(f"missing={missing:.2f}")
    return {"blocked": bool(reasons), "psi": drift, "missing": missing, "reasons": reasons}


def row_codes(df):
    """Первое сработавшее правило определяет отказ; порядок совпадает с приоритетом комплаенса."""
    code = pd.Series("", index=df.index, dtype=object)
    bad = df[FEATURES_V2].isna().any(axis=1) | ~df.industry.isin(INDUSTRIES) | df.revenue_year.isna()
    for col, (low, high) in RANGES.items():
        bad |= (df[col] < low) | (df[col] > high)
    checks = [("REJECT_SCHEMA", bad),
              ("EXCLUDE_COMPLIANCE", df.compliance.astype(bool)), ("EXCLUDE_LIQUIDATION", df.liquidation.astype(bool)),
              ("EXCLUDE_TOCHKA", df.tochka.astype(bool)), ("EXCLUDE_CONSENT", ~df.consent.astype(bool)),
              ("EXCLUDE_FREQUENCY", df.days_last_offer < 30),
              ("EXCLUDE_SEGMENT", (df.tenure < 6) | ~df.tariff.isin(["advanced", "professional"])),
              ("ESCALATE_VALUE", df.revenue_year > REVENUE_LIMIT),
              ("HOLD_SEASONAL", df.industry.isin(SEASONAL) & (df.last_year_forward_ratio < SEASONAL_DROP))]
    for name, mask in checks:
        code = code.mask((code == "") & mask.fillna(False).astype(bool), name)
    return code


def decide(df, bundle, reference, budget=3_000_000, seed=20261101, threshold=None):
    """Возвращает решение по каждому клиенту и итог партии."""
    missing = sorted(set(FEATURES_V2 + FLAGS + EXTRA) - set(df.columns))
    if missing:
        raise SchemaError(f"Missing columns: {missing}")
    gate = batch_gate(df, reference)
    out = pd.DataFrame({"client_id": df.client_id.to_numpy()}, index=df.index)
    code = row_codes(df)
    out["reason_code"] = code
    for col in ["risk", "offer", "budget_cost", "saving_year", "predicted_uplift"]:
        out[col] = 0.
    out["offer_name"] = ""
    out["group"] = ""
    usable = df[code == ""]
    if len(usable) == 0:
        out["decision"] = np.where(out.reason_code.str.startswith("ESCALATE") | out.reason_code.eq("HOLD_SEASONAL"), "MANAGER", "NONE")
        return out, gate
    risk = bundle["v2s"].predict(usable)["any"]
    out.loc[usable.index, "risk"] = risk
    if threshold is None:
        threshold = float(np.quantile(bundle["v2s"].predict(reference.assign(client_id=""))["any"], .9))
    uplift = bundle["uplift"].predict(usable)
    q = quote_vec(usable.tariff, usable.payments)
    revenue = usable.revenue_year.to_numpy()
    net = np.stack([np.where(q[a]["allowed"], uplift[:, a - 1] * revenue * MARGIN - q[a]["cost"], -np.inf) for a in [1, 2, 3]], axis=1)
    allowed_any = np.stack([q[a]["allowed"] for a in [1, 2, 3]], axis=1).any(axis=1)
    best = net.argmax(axis=1)
    offer = best + 1
    index = np.arange(len(usable))
    best_uplift = uplift[index, best]
    cost = np.stack([q[a]["cost"] for a in [1, 2, 3]], axis=1)[index, best].astype(float)
    saving = np.stack([q[a]["saving"] for a in [1, 2, 3]], axis=1)[index, best].astype(float)
    labels = np.where(~allowed_any, "NO_SAFE_OFFER", np.where(net[index, best] <= 0, "NO_OFFER_NEGATIVE_EFFECT",
                      np.where(risk < threshold, "NO_ACTION_LOW_RISK", "CANDIDATE")))
    out.loc[usable.index, "reason_code"] = labels
    candidates = pd.DataFrame({"gain": net[index, best], "cost": cost, "offer": offer, "saving": saving, "uplift": best_uplift,
                               "client_id": usable.client_id.to_numpy()}, index=usable.index)[labels == "CANDIDATE"]
    candidates = candidates.sort_values(["gain", "client_id"], ascending=[False, True])
    reserved = candidates.cost.cumsum()
    chosen = candidates[reserved <= budget]
    out.loc[candidates.index.difference(chosen.index), "reason_code"] = "BUDGET_EXHAUSTED"
    rng = np.random.default_rng(seed)
    control = np.zeros(len(chosen), bool)
    control[rng.permutation(len(chosen))[:int(np.ceil(.1 * len(chosen)))]] = True
    for col, name in [("offer", "offer"), ("budget_cost", "cost"), ("saving_year", "saving"), ("predicted_uplift", "uplift")]:
        out.loc[chosen.index, col] = chosen[name].to_numpy()
    out.loc[chosen.index, "offer_name"] = chosen.offer.map(OFFERS).to_numpy()
    out.loc[chosen.index, "group"] = np.where(control, "holdout", "treatment")
    out.loc[chosen.index, "reason_code"] = np.where(control, "HOLDOUT", "OFFER")
    if gate["blocked"]:
        # Отказ модели на партии: автоматики нет, менеджер получает список правил Gate 3.
        out.loc[out.reason_code.isin(["OFFER", "HOLDOUT"]), ["offer", "budget_cost", "saving_year"]] = 0.
        out.loc[out.reason_code.isin(["OFFER", "HOLDOUT"]), "group"] = ""
        out.loc[out.reason_code.isin(["OFFER", "HOLDOUT"]), "offer_name"] = ""
        out["reason_code"] = out.reason_code.replace({"OFFER": "BLOCKED_DRIFT", "HOLDOUT": "BLOCKED_DRIFT"})
        out.loc[(out.reason_code == "BLOCKED_DRIFT") | ((out.index.isin(usable.index)) & (rule_score(df) > 0)), "manual_rules"] = True
    out["decision"] = np.select(
        [out.reason_code.eq("OFFER"), out.reason_code.eq("HOLDOUT"),
         out.reason_code.isin(["HOLD_SEASONAL", "ESCALATE_VALUE", "BLOCKED_DRIFT"])],
        ["SEND", "CONTROL", "MANAGER"], "NONE")
    return out, gate


def revalidate(decisions, fresh):
    """Перед отправкой флаги перечитываются: задержка флага не должна доходить до клиента."""
    rows = decisions.client_id.isin(fresh.loc[fresh.compliance | fresh.liquidation | ~fresh.consent, "client_id"])
    result = decisions.copy()
    result.loc[rows & result.decision.eq("SEND"), ["decision", "reason_code"]] = ["NONE", "EXCLUDE_REVALIDATION"]
    return result
