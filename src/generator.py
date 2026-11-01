"""Панель v2. Скрытые драйверы, потенциальные исходы и эталон не передаются моделям."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (AMPLITUDE, CAL_CUTS, HIBERNATE, INDUSTRIES, INDUSTRY_P, MONTHS, PEAK, REASONS,
                    TEST_CUTS, TRAIN_CUTS, Scenario)
from legacy import quote_vec

DAYS_IN_MONTH = 30.4
LABEL_COLS = ["y_close", "y_quiet", "y_quiet_adj", "y_any", "y_any_adj"]


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x)))


def bisect(f, low, high, steps=50):
    for _ in range(steps):
        mid = (low + high) / 2
        low, high = (mid, high) if f(mid) < 0 else (low, mid)
    return (low + high) / 2


def season_matrix():
    """Индустрия × месяц года."""
    moy = np.arange(12)
    return np.stack([1 + a * np.cos(2 * np.pi * (moy - p) / 12) for a, p in zip(AMPLITUDE, PEAK)])


def off_season_matrix():
    moy = np.arange(12)
    distance = [np.minimum((moy - p) % 12, (p - moy) % 12) for p in PEAK]
    return np.stack(distance) >= 4


def event_labels(panel, cut, close_event):
    """Правило Gate 2 и его сезонная версия считаются только по панели."""
    n = len(cut)
    idx = np.arange(n)
    quiet, recurring = np.zeros(n, bool), np.zeros(n, bool)
    for shift in [0, 1]:
        k = cut + shift
        ref = np.stack([panel[idx, k - j] for j in [1, 2, 3]], axis=1).mean(axis=1)
        new_event = ~((panel[idx, k - 1] < .3 * ref) & (panel[idx, k - 2] < .3 * ref))
        hit = (panel[idx, k] < .3 * ref) & (panel[idx, k + 1] < .3 * ref) & new_event
        quiet |= hit
        last_ref = np.stack([panel[idx, k - 12 - j] for j in [1, 2, 3]], axis=1).mean(axis=1)
        recurring |= hit & (panel[idx, k - 12] < .3 * last_ref) & (panel[idx, k - 11] < .3 * last_ref)
    return quiet, quiet & ~recurring | (close_event & quiet)


def generate(s: Scenario):
    rng = np.random.default_rng(s.seed)
    n, T = s.n, len(MONTHS)
    ntrain, ncal = int(n * .36), int(n * .14)
    split = np.array(["train"] * ntrain + ["calibration"] * ncal + ["test"] * (n - ntrain - ncal))
    cut = np.r_[rng.choice(TRAIN_CUTS, ntrain), rng.choice(CAL_CUTS, ncal), rng.choice(TEST_CUTS, n - ntrain - ncal)]
    idx = np.arange(n)

    industry = rng.choice(len(INDUSTRIES), n, p=INDUSTRY_P)
    region = rng.integers(0, 3, n)
    tariff = rng.choice(["advanced", "professional"], n, p=[.8, .2])
    payments = rng.integers(15, 181, n)
    tenure = rng.integers(3, 73, n)
    services = rng.integers(0, 5, n)
    cash = rng.beta(2, 2, n)

    # Намерение и внешняя причина не наблюдаются: признаки служат лишь их шумными следами.
    intent = rng.standard_normal(n)
    external = rng.standard_normal(n)
    shock = rng.random(n) < sigmoid(-1.6 + .5 * intent)
    share = np.clip(rng.beta(1.3, 3, n) + .06 * intent, 0, 1)
    complaints = rng.poisson(np.exp(-1.2 + .35 * intent + .6 * shock))
    fee_burden = .15 + .5 * shock + rng.beta(1.5, 3, n)

    base = rng.lognormal(14, .65, n).astype("float32")
    scale = (np.log(base) - 14) / .65
    raw_revenue = np.exp(.45 * (.7 * scale + .714 * rng.standard_normal(n)))
    revenue = 60_000 * raw_revenue / raw_revenue.mean()

    season = season_matrix()
    moy = MONTHS.month.to_numpy() - 1
    panel = base[:, None] * season[industry][:, moy].astype("float32")
    panel *= np.exp(.10 * rng.standard_normal((n, T), dtype="float32"))

    hibernator = rng.random(n) < np.array(HIBERNATE)[industry]
    off = off_season_matrix()
    rows = np.where(hibernator)[0]
    panel[rows] *= np.where(off[industry[rows]][:, moy], rng.uniform(.08, .2, (len(rows), T)), 1).astype("float32")

    declining = rng.random(n) < sigmoid(-1.9 + 1.2 * intent)
    blip = rng.random(n) < .03
    panel[idx, cut - 2] *= np.where(declining, .82, 1).astype("float32")
    panel[idx, cut - 1] *= np.where(declining, .52, 1).astype("float32") * np.where(blip, .45, 1).astype("float32")
    zero = rng.random(n) < .018
    panel[idx[zero], cut[zero] - 1] = 0

    # Время до ухода без воздействия; интенсивность подбирается под заданный отток.
    log_h = 1.1 * intent + .45 * shock - .22 * services + .7 * (fee_burden - fee_burden.mean()) + s.latent_sigma * external
    level = bisect(lambda a: (1 - np.exp(-np.exp(a + log_h) * 365)).mean() - s.annual_churn, -20, 0)
    daily = np.exp(level + log_h)
    t0 = rng.exponential(1 / daily)

    # Причина ухода: сдвиг неуправляемых причин подбирается под заданную долю управляемых.
    price = np.stack([1.2 * shock + 1.5 * (fee_burden - .6),
                      1.0 * (payments < 40) + .5 * (tariff == "professional")], axis=1)
    other = np.stack([2.2 * (share - .3), .5 * complaints, np.zeros(n)], axis=1)
    weight = 1 - np.exp(-daily * 365)

    def manageable_gap(shift):
        utility = np.exp(np.c_[price, other + shift])
        p = utility / utility.sum(axis=1, keepdims=True)
        return s.manageable_share - np.average(p[:, :2].sum(axis=1), weights=weight)

    shift = bisect(manageable_gap, -8, 8)
    utility = np.exp(np.c_[price, other + shift])
    p = utility / utility.sum(axis=1, keepdims=True)
    reason = (p.cumsum(axis=1) < rng.random((n, 1))).sum(axis=1).clip(0, 4)

    forced = rng.random(n) < s.hidden_forced
    t_forced = rng.uniform(0, 60, n)
    t0 = np.where(forced, np.minimum(t0, t_forced), t0)
    reason = np.where(forced, 5, reason)

    # Потенциальные исходы для контроля и трёх офферов на общих случайных числах.
    quotes = quote_vec(tariff, payments)
    match = np.array([[1, .2, .03, .03, .03, 0], [.3, 1, .03, .03, .03, 0], [.5, .5, .03, .03, .03, 0]])
    u_save, u_dog = rng.random(n), rng.random(n)
    delay = rng.exponential(s.persistence_days, n)
    dog_time = rng.uniform(0, 90, n)
    dog_p = s.dog_base + s.dog_extra * ((services == 0) & (complaints > 0))
    times = {0: t0}
    for offer in [1, 2, 3]:
        fit = np.where(cash > .4, 1, .2) if offer == 3 else 1.
        saved = quotes[offer]["allowed"] & (u_save < s.save_rate * match[offer - 1, reason] * fit)
        t = t0 + saved * delay
        times[offer] = np.where((u_dog < dog_p) & (t0 > 365), dog_time, t)
    churn60 = {a: times[a] <= 60 for a in times}
    churn365 = {a: times[a] <= 365 for a in times}
    arm = rng.integers(0, 4, n)
    retained60 = ~np.choose(arm, [churn60[a] for a in range(4)])
    retained365 = ~np.choose(arm, [churn365[a] for a in range(4)])

    # Метки считаются по панели мира без воздействия.
    close_type = rng.random(n) < .4
    start = cut + np.floor(t0 / DAYS_IN_MONTH).astype(np.int64)
    for k in np.unique(start[start < T]):
        rows = np.where(start == k)[0]
        closed = close_type[rows] | forced[rows]
        ref = panel[rows[~closed], k - 3:k].mean(axis=1)
        panel[rows[~closed], k:] = (ref[:, None] * rng.uniform(.08, .24, (len(ref), T - k))).astype("float32")
        panel[rows[closed], k:] = 0
    y_close = (t0 <= 60) & close_type & ~forced
    y_quiet, y_quiet_adj = event_labels(panel, cut, t0 <= 60)

    df = pd.DataFrame({
        "client_id": [f"SYN-{i:06d}" for i in idx], "split": split, "cut": cut.astype("int16"),
        "cutoff": MONTHS[cut], "industry": np.array(INDUSTRIES)[industry],
        "region": np.array(["A", "B", "C"])[region], "tariff": tariff, "payments": payments,
        "tenure": tenure, "services": services, "cash_buffer": cash, "fee_shock": shock.astype(int),
        "fee_burden": fee_burden, "self_transfer_share": share, "complaints": complaints,
        "days_no_incoming": np.where(zero, rng.integers(30, 60, n), rng.integers(0, 12, n)),
        "compliance": rng.random(n) < .025, "liquidation": rng.random(n) < .012, "tochka": rng.random(n) < .018,
        "consent": rng.random(n) > .08, "days_last_offer": rng.integers(0, 180, n), "revenue_year": revenue,
        "arm": arm, "retained60": retained60.astype(int), "retained365": retained365.astype(int),
        "y_close": y_close.astype(int), "y_quiet": y_quiet.astype(int), "y_quiet_adj": y_quiet_adj.astype(int),
    })
    df["y_any"] = np.maximum(df.y_close, df.y_quiet)
    df["y_any_adj"] = np.maximum(df.y_close, df.y_quiet_adj)
    truth = pd.DataFrame({
        "t0": t0, "true_churn60": t0 <= 60, "true_churn365": t0 <= 365, "hibernator": hibernator,
        "forced_hidden": forced, "reason": np.array(REASONS)[reason], "close_days": np.where(y_close, t0, np.nan),
        **{f"keep60_{a}": ~churn60[a] for a in range(4)}, **{f"keep365_{a}": ~churn365[a] for a in range(4)},
    })
    return panel, df, truth


def season_profile(panel, industry):
    """Профиль по 2021 году: период раньше всех срезов, признаки не заглядывают вперёд."""
    first = panel[:, :12]
    normal = first / np.maximum(first.mean(axis=1, keepdims=True), 1)
    frame = pd.DataFrame(normal)
    return frame.groupby(np.asarray(industry)).median().reindex(INDUSTRIES).to_numpy()


def add_features(panel, df, profile=None):
    """Все признаки используют только месяцы строго до среза."""
    cut, idx = df.cut.to_numpy(), np.arange(len(df))
    if profile is None:
        train = df.split.eq("train").to_numpy()
        profile = season_profile(panel[train], df.industry[train])
    ind = df.industry.map({k: i for i, k in enumerate(INDUSTRIES)}).to_numpy()
    moy = MONTHS.month.to_numpy() - 1
    at = lambda lag: panel[idx, cut - lag]
    older = (at(2) + at(3) + at(4)) / 3
    df["turnover_ratio"] = at(1) / np.maximum(older, 1)
    expected = profile[ind, moy[cut - 1]] / np.stack([profile[ind, moy[cut - j]] for j in [2, 3, 4]], axis=1).mean(axis=1)
    df["season_adjusted_ratio"] = df.turnover_ratio / expected
    df["yoy_ratio"] = at(1) / np.maximum(at(13), 1)
    df["yoy_ratio_3m"] = (at(1) + at(2) + at(3)) / np.maximum(at(13) + at(14) + at(15), 1)
    df["last_year_forward_ratio"] = at(12) / np.maximum((at(13) + at(14) + at(15)) / 3, 1)
    for col in ["turnover_ratio", "season_adjusted_ratio", "yoy_ratio", "yoy_ratio_3m", "last_year_forward_ratio"]:
        df[col] = df[col].astype("float32").clip(0, 10)
    return df, profile


def split_cuts_are_valid():
    """Срезы и созревание меток не пересекаются между частями."""
    available = lambda t: MONTHS[t] + pd.Timedelta(days=120)
    return (available(max(TRAIN_CUTS)) < MONTHS[min(CAL_CUTS)]
            and available(max(CAL_CUTS)) < MONTHS[min(TEST_CUTS)]
            and MONTHS[max(TRAIN_CUTS)] + pd.Timedelta(days=365) < MONTHS[min(TEST_CUTS)])
