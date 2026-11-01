"""Разведочный анализ после финального запуска: не входил в предрегистрацию."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import COHORT_SELECTED, DATA, MARGIN, OPS_PER_COHORT, RESULTS, TREATED_SHARE
from experiment import dump
from legacy import quote_vec, rule_score
from metrics import policy_value, power_total, summary, top_mask


def main():
    test = pd.read_pickle(DATA / "test.pkl")
    month = test.cut.to_numpy()
    revenue = test.revenue_year.to_numpy()
    rule_rank = rule_score(test) + np.random.default_rng(20261101).random(len(test)) * 1e-3
    adjusted = {name: summary(test.y_any_adj.to_numpy(), score, month, prob) for name, score, prob in
                [("rules", rule_rank, test.risk_rule.to_numpy()), ("catboost_v2s", test.risk_s.to_numpy(), test.risk_s.to_numpy()),
                 ("catboost_v2", test.risk_v2.to_numpy(), test.risk_v2.to_numpy())]}
    keep365 = test[[f"keep365_{a}" for a in range(4)]].to_numpy()
    keep60 = test[[f"keep60_{a}" for a in range(4)]].to_numpy()
    q = quote_vec(test.tariff, test.payments)
    package = (q[1]["allowed"] & (q[1]["saving"] > 0)).astype(int)
    cost = np.full(len(test), 2970.)
    treated = COHORT_SELECTED * TREATED_SHARE
    pivot, required = {}, {}
    value = test.risk_s.to_numpy() * revenue
    for depth in [.01, .05, .1]:
        mask = top_mask(np.where(package > 0, value, -np.inf), month, depth) & (package > 0)
        pivot[str(depth)] = {f"alpha_{a}": policy_value(mask, package, cost, revenue, keep365, keep60, alpha=a) for a in [1., .5, .3, .1]}
        base = pivot[str(depth)]["alpha_1.0"]
        gain = base["uplift12_pp"] / 100 * base["mean_revenue"]
        required[str(depth)] = {"gain_rub_per_treated": gain, "max_cost_breakeven": gain * MARGIN - OPS_PER_COHORT / treated,
                                "max_cost_3rub": gain / 3, "keep365_control": float(keep365[mask, 0].astype(float).mean()),
                                "true_churn365_pct": float(test.true_churn365[mask].astype(float).mean() * 100)}
    top = top_mask(test.risk_s.to_numpy(), month, .1)
    power = {}
    for depth, row in required.items():
        u = pivot[depth]["alpha_1.0"]["uplift12_pp"] / 100
        p0 = row["keep365_control"]
        power[depth] = {"control_retention": p0, "effect_pp": u * 100, "n_90_10": power_total(p0, u), "n_50_50": power_total(p0, u, .5)}
    main_result = json.loads((RESULTS / "main.json").read_text())
    base = main_result["policies"]["0.1"]["risk_adj"]
    clients = 700_000
    scale = {"cohort_clients": int(clients * .1), "treated": int(clients * .1 * TREATED_SHARE),
             "extra_revenue_mln": clients * .1 * TREATED_SHARE * base["uplift12_pp"] / 100 * base["mean_revenue"] / 1e6,
             "offer_cost_mln": clients * .1 * TREATED_SHARE * base["mean_cost"] / 1e6}
    scale["margin_minus_cost_mln"] = scale["extra_revenue_mln"] * MARGIN - scale["offer_cost_mln"]
    oracle = main_result["policies"]["0.1"]["oracle_bound"]
    result = {"adjusted_target": adjusted, "package_only_top_by_risk_x_revenue": pivot, "required_action_cost": required,
              "power_selected_group": power, "scale_base_700k": scale,
              "oracle_share_of_admitted_pct": oracle["selected"] / len(test) * 100,
              "top10_true_churn365_pct": float(test.true_churn365[top].astype(float).mean() * 100)}
    dump("analysis.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=1, default=float)[:3800])


if __name__ == "__main__":
    main()
