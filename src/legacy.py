"""Детерминированные правила, перенесённые из прототипа Gate 3 без изменения логики."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARIFFS = {
    "simple": {"base": 490, "included": 5, "payment": 49},
    "advanced": {"base": 1990, "included": 10, "payment": 29},
    "professional": {"base": 4990, "included": 100, "payment": 19},
}
OFFERS = {1: "Пакет комиссий", 2: "Тариф ниже", 3: "Годовая оплата"}


def monthly_cost(tariff, payments):
    t = TARIFFS[tariff]
    return int(t["base"] + max(0, int(payments) - t["included"]) * t["payment"])


def quote(tariff, payments, offer):
    """Калькулятор: гардрейл проверяет и первый год, и регулярный месяц после промо."""
    current = monthly_cost(tariff, payments)
    t = TARIFFS[tariff]
    if offer == 1:
        saved_fees = min(max(int(payments) - t["included"], 0), 40) * t["payment"]
        recurring = current - saved_fees + 990
        after_year = 12 * (current - saved_fees) + 9 * 990
        cost = 2970
    elif offer == 2:
        lower = "advanced" if tariff == "professional" else "simple"
        recurring = monthly_cost(lower, payments)
        after_year = 12 * recurring
        cost = max(0, 12 * (current - recurring))
    elif offer == 3:
        discount = round(t["base"] * 12 * .17)
        after_year = 12 * current - discount
        recurring = after_year / 12
        cost = discount
    else:
        raise ValueError("Unknown offer")
    return {"current_month": current, "after_month": recurring, "before_year": 12 * current,
            "after_year": int(after_year), "saving_year": int(12 * current - after_year),
            "budget_cost": int(cost), "allowed": bool(after_year <= 12 * current and recurring <= current),
            "offer": int(offer)}


def quote_vec(tariff, payments):
    """Векторный расчёт для трёх офферов. Совпадение с quote проверяется тестом."""
    prof = np.asarray(tariff) == "professional"
    pay = np.asarray(payments, dtype=np.int64)
    base = np.where(prof, 4990, 1990)
    included = np.where(prof, 100, 10)
    fee = np.where(prof, 19, 29)
    current = base + np.maximum(0, pay - included) * fee
    saved = np.minimum(np.maximum(pay - included, 0), 40) * fee
    out = {}
    recurring = current - saved + 990
    after = 12 * (current - saved) + 9 * 990
    out[1] = {"recurring": recurring, "after": after, "cost": np.full(len(pay), 2970)}
    low_base, low_included, low_fee = np.where(prof, 1990, 490), np.where(prof, 10, 5), np.where(prof, 29, 49)
    recurring = low_base + np.maximum(0, pay - low_included) * low_fee
    out[2] = {"recurring": recurring, "after": 12 * recurring, "cost": np.maximum(0, 12 * (current - recurring))}
    discount = np.floor(base * 12 * .17 + .5).astype(np.int64)
    after = 12 * current - discount
    out[3] = {"recurring": after / 12, "after": after, "cost": discount}
    for offer in out.values():
        offer["allowed"] = (offer["after"] <= 12 * current) & (offer["recurring"] <= current)
        offer["saving"] = 12 * current - offer["after"]
    return out


def eligibility(df):
    return (~df.compliance.astype(bool) & ~df.liquidation.astype(bool) & ~df.tochka.astype(bool)
            & df.consent.astype(bool) & (df.days_last_offer >= 30) & (df.tenure >= 6)
            & df.tariff.isin(["advanced", "professional"]))


def rule_score(df):
    return (df.days_no_incoming.ge(30).astype(int) * 2 + df.turnover_ratio.lt(.5).astype(int)).to_numpy(float)


def psi(reference, current):
    reference, current = np.asarray(reference), np.asarray(current)
    bins = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    bins[0], bins[-1] = -np.inf, np.inf
    a = np.histogram(reference, bins)[0].astype(float)
    b = np.histogram(current, bins)[0].astype(float)
    a = np.maximum(a / a.sum(), 1e-5)
    b = np.maximum(b / b.sum(), 1e-5)
    return float(np.sum((b - a) * np.log(b / a)))
