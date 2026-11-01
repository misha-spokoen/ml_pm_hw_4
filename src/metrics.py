"""Метрики под бюджет. Верхние k считаются внутри месяца, как в реальной кампании."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss

from config import COHORT_SELECTED, DEV_COST, MARGIN, OPS_PER_COHORT, TREATED_SHARE


def top_mask(score, month, frac=.1, seed=20261101):
    """Равные скоры разрешает фиксированный случайный порядок, не зависящий от меток."""
    score, month = np.asarray(score, float), np.asarray(month)
    jitter = np.random.default_rng(seed).random(len(score))
    mask = np.zeros(len(score), bool)
    for m in np.unique(month):
        rows = np.where(month == m)[0]
        k = int(np.ceil(frac * len(rows)))
        order = np.lexsort((jitter[rows], -score[rows]))
        mask[rows[order[:k]]] = True
    return mask


def summary(y, score, month, probability=None, frac=.1):
    y = np.asarray(y)
    mask = top_mask(score, month, frac)
    precision = float(y[mask].mean())
    out = {"n": int(len(y)), "positive": int(y.sum()), "base_rate": float(y.mean()), "top_n": int(mask.sum()),
           "precision": precision, "lift": precision / float(y.mean()), "recall": float(y[mask].sum() / y.sum())}
    if probability is not None:
        out["brier"] = float(brier_score_loss(y, probability))
    return out


def boot_precision_diff(y, mask_a, mask_b, reps=400, seed=20261101):
    """Парный бутстрэп разности точности по клиентам, попавшим хотя бы в один список."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    rows = np.where(mask_a | mask_b)[0]
    ya, a, b = y[rows], mask_a[rows], mask_b[rows]
    out = []
    for _ in range(reps):
        s = rng.integers(0, len(rows), len(rows))
        out.append(ya[s][a[s]].mean() - ya[s][b[s]].mean())
    return [float(x) for x in np.quantile(out, [.025, .975])]


def group_recall(y, mask, groups):
    groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        rows = groups == g
        out[str(g)] = {"n": int(rows.sum()), "positive": int(y[rows].sum()), "recall": float(y[rows & mask].sum() / max(y[rows].sum(), 1))}
    values = [v["recall"] for v in out.values()]
    return out, float(max(values) - min(values))


def gap_tests(y, mask, groups, reps=2000, seed=1):
    """Отделяет шум от смещения: перестановка групп и бутстрэп разрыва."""
    rng = np.random.default_rng(seed)
    y, groups = np.asarray(y), np.asarray(groups)
    positive = np.where(y == 1)[0]
    hit = mask[positive].astype(float)
    labels, codes = np.unique(groups[positive], return_inverse=True)
    counts = np.bincount(codes, minlength=len(labels))

    def gap(c, h):
        rates = np.bincount(c, weights=h, minlength=len(labels)) / np.maximum(np.bincount(c, minlength=len(labels)), 1)
        return rates.max() - rates.min()

    observed = gap(codes, hit)
    null = np.array([gap(rng.permutation(codes), hit) for _ in range(reps)])
    boots = []
    for _ in range(reps):
        s = rng.integers(0, len(positive), len(positive))
        boots.append(gap(codes[s], hit[s]))
    return {"gap": float(observed), "permutation_p": float((null >= observed).mean()),
            "null_95": float(np.quantile(null, .95)), "null_share_over_5pp": float((null > .05).mean()),
            "bootstrap_95": [float(x) for x in np.quantile(boots, [.025, .975])],
            "positive_min": int(counts.min())}


def wilson(k, n, z=1.96):
    if n == 0:
        return [0., 1.]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def policy_value(mask, offer, cost, revenue, keep365, keep60, alpha=1.):
    """Истинный эффект по скрытым потенциальным исходам, в ожидании по случайному контролю 10%."""
    rows = np.where(mask & (offer > 0))[0]
    if len(rows) == 0:
        return {"selected": 0}
    a = offer[rows]
    u12 = keep365[rows, a].astype(float) - keep365[rows, 0].astype(float)
    u60 = keep60[rows, a].astype(float) - keep60[rows, 0].astype(float)
    rev = revenue[rows]
    gain = float((u12 * rev).mean())
    spend = float(cost[rows].mean()) * alpha
    treated = COHORT_SELECTED * TREATED_SHARE
    out = {"selected": int(len(rows)), "uplift12_pp": float(u12.mean() * 100), "uplift60_pp": float(u60.mean() * 100),
           "mean_revenue": float(rev.mean()), "mean_cost": float(cost[rows].mean()), "alpha": alpha,
           "revenue_per_budget": gain / spend if spend else None,
           "margin_per_1000": treated * (gain * MARGIN - spend),
           "after_ops_per_1000": treated * (gain * MARGIN - spend) - OPS_PER_COHORT,
           "first_launch_per_1000": treated * (gain * MARGIN - spend) - OPS_PER_COHORT - DEV_COST,
           "breakeven_pp": (spend + OPS_PER_COHORT / treated) / (float(rev.mean()) * MARGIN) * 100,
           "offer_mix": {str(k): float((a == k).mean()) for k in [1, 2, 3]}}
    return out


def power_total(p0, effect, control=.1, alpha_z=1.959964, power_z=.841621):
    p1 = p0 + effect
    variance = p0 * (1 - p0) / control + p1 * (1 - p1) / (1 - control)
    return float((alpha_z + power_z) ** 2 * variance / effect ** 2)
