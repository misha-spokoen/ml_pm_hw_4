"""Основной запуск: панель v2, модели, сравнение с правилами, срезы, политики, экономика."""
from __future__ import annotations

import gc
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

from config import (DATA, DEV, MODELS, RESULTS, S0, SEASONAL, SENSITIVITY, Scenario)
from generator import add_features, off_season_matrix, generate, split_cuts_are_valid
from config import INDUSTRIES
from legacy import eligibility, psi, quote_vec, rule_score
from metrics import (boot_precision_diff, gap_tests, group_recall, policy_value, power_total, summary, top_mask)
from models import FEATURES_G3, FEATURES_V2, LogisticBaseline, RiskModel, RuleBaseline, UpliftModel, assert_no_forbidden


def dump(name, obj):
    RESULTS.mkdir(exist_ok=True)
    def scalar(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.bool_,)): return bool(x)
        raise TypeError(type(x))
    (RESULTS / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=scalar))


def rule_offer(df):
    """Матрица «причина → действие» без ML: сначала самое дешёвое из допустимых."""
    q = quote_vec(df.tariff, df.payments)
    offer = np.zeros(len(df), int)
    offer = np.where(q[1]["allowed"] & (q[1]["saving"] > 0), 1, offer)
    offer = np.where((offer == 0) & q[2]["allowed"] & (q[2]["saving"] > 0), 2, offer)
    offer = np.where((offer == 0) & q[3]["allowed"] & (df.cash_buffer.to_numpy() > .4), 3, offer)
    cost = np.select([offer == 1, offer == 2, offer == 3], [q[1]["cost"], q[2]["cost"], q[3]["cost"]], 0)
    return offer, cost.astype(float), q


def uplift_policy(df, uplift, q, margin=.5):
    """Лучший безопасный оффер по ожидаемой марже; уход в минус означает отказ от контакта."""
    revenue = df.revenue_year.to_numpy()
    value = np.full((len(df), 4), -np.inf)
    value[:, 0] = 0
    for a in [1, 2, 3]:
        net = uplift[:, a - 1] * revenue * margin - q[a]["cost"]
        value[:, a] = np.where(q[a]["allowed"], net, -np.inf)
    offer = value.argmax(axis=1)
    cost = np.select([offer == 1, offer == 2, offer == 3], [q[1]["cost"], q[2]["cost"], q[3]["cost"]], 0).astype(float)
    predicted = np.where(offer > 0, uplift[np.arange(len(df)), np.maximum(offer, 1) - 1], 0.) * revenue
    return offer, cost, predicted


def risk_table(y, month, scores):
    """scores: имя → (скор, вероятность)."""
    return {k: summary(y, s, month, p) for k, (s, p) in scores.items()}


def seasonal_checks(model, test, risk, top):
    """C4.2b: истинно стабильные клиенты в низкий и высокий сезон; C4.2a: парный контрфакт."""
    off = off_season_matrix()
    moy = (test.cutoff.dt.month.to_numpy() - 1)
    ind = test.industry.map({k: i for i, k in enumerate(INDUSTRIES)}).to_numpy()
    low = off[ind, moy]
    stable = test.t0.to_numpy() > 120
    seasonal = test.industry.isin(SEASONAL).to_numpy()
    calendar = {}
    for name in SEASONAL:
        rows = stable & (test.industry == name).to_numpy()
        calendar[name] = {"mean_low_pct": float(risk[rows & low].mean() * 100), "mean_high_pct": float(risk[rows & ~low].mean() * 100),
                          "top_share_low_pct": float(top[rows & low].mean() * 100), "top_share_high_pct": float(top[rows & ~low].mean() * 100)}
        calendar[name]["rise_pp"] = calendar[name]["mean_low_pct"] - calendar[name]["mean_high_pct"]
    base = float(top[stable & ~seasonal].mean() * 100)
    worst = max(v["top_share_low_pct"] for v in calendar.values())
    hib = test.hibernator.to_numpy()
    grid = []
    sample = test.sample(min(400, len(test)), random_state=1)
    for step in [.9, .7, .5]:
        a = sample.copy(); b = sample.copy()
        a["turnover_ratio"] = 1.; b["turnover_ratio"] = step
        for x in (a, b):
            x["season_adjusted_ratio"] = 1.; x["yoy_ratio"] = 1.; x["yoy_ratio_3m"] = 1.; x["days_no_incoming"] = 0
        delta = model.predict(b)["any"] - model.predict(a)["any"]
        grid.append({"step": step, "mean_rise_pp": float(delta.mean() * 100), "max_rise_pp": float(delta.max() * 100)})
    return {"calendar": calendar, "top_share_non_seasonal_pct": base, "worst_low_ratio": worst / base if base else None,
            "c4_2a_grid": grid, "c4_2a_pass": max(g["max_rise_pp"] for g in grid) <= .1,
            "c4_2b_pass": max(v["rise_pp"] for v in calendar.values()) <= .5 and worst <= 1.5 * base,
            "hibernator_top_share_pct": float(top[hib].mean() * 100) if hib.any() else None}


def run(s: Scenario, full=True):
    started = time.time()
    panel, df, truth = generate(s)
    df, profile = add_features(panel, df)
    del panel
    gc.collect()
    df = pd.concat([df, truth], axis=1)
    del truth
    admitted = eligibility(df)
    funnel = {"clients": len(df), "admitted": int(admitted.sum()), "by_split": df[admitted].split.value_counts().to_dict(),
              "excluded": {"compliance": int(df.compliance.sum()), "liquidation": int(df.liquidation.sum()), "tochka": int(df.tochka.sum()),
                           "no_consent": int((~df.consent).sum()), "recent_offer": int((df.days_last_offer < 30).sum()), "short_tenure": int((df.tenure < 6).sum())}}
    data = df[admitted].copy()
    del df
    gc.collect()
    train, cal, test = [data[data.split == k].copy() for k in ["train", "calibration", "test"]]
    for features in (FEATURES_V2, FEATURES_G3):
        assert_no_forbidden(features)

    v2 = RiskModel(FEATURES_V2).fit(train, cal)
    v2s = RiskModel(FEATURES_V2, "y_quiet_adj", "y_any_adj").fit(train, cal)
    rules = RuleBaseline().fit(cal)
    risk_v2 = v2.predict(test)
    risk_s = v2s.predict(test)
    p_rule = rules.predict(test)
    rule_rank = rule_score(test) + np.random.default_rng(s.seed).random(len(test)) * 1e-3
    month = test.cut.to_numpy()
    y = test.y_any.to_numpy()
    true60 = test.true_churn60.to_numpy().astype(int)

    if full:
        g3 = RiskModel(FEATURES_G3).fit(train, cal)
        logistic = LogisticBaseline().fit(train, cal)
        risk_g3 = g3.predict(test)
        p_log = logistic.predict(test)
    uplift_model = UpliftModel().fit(train)
    uplift = uplift_model.predict(test)

    offer_rule, cost_rule, q = rule_offer(test)
    offer_up, cost_up, predicted = uplift_policy(test, uplift, q)
    revenue = test.revenue_year.to_numpy()
    keep365 = test[[f"keep365_{a}" for a in range(4)]].to_numpy()
    keep60 = test[[f"keep60_{a}" for a in range(4)]].to_numpy()
    true_gain = (keep365[:, 1:].astype(float) - keep365[:, [0]].astype(float)) * revenue[:, None]

    results = {"scenario": s.__dict__, "funnel": funnel}
    scores = {"rules": (rule_rank, p_rule), "catboost_v2": (risk_v2["any"], risk_v2["any"]),
              "catboost_v2s": (risk_s["any"], risk_s["any"])}
    if full:
        scores["logistic_v2"] = (p_log, p_log)
        scores["catboost_g3_features"] = (risk_g3["any"], risk_g3["any"])
    results["risk"] = {"official_target": risk_table(y, month, scores), "true_churn60": risk_table(true60, month, scores),
                       "close": summary(test.y_close.to_numpy(), risk_v2["close"], month, risk_v2["close"]),
                       "quiet": summary(test.y_quiet.to_numpy(), risk_v2["quiet"], month, risk_v2["quiet"])}
    model_top = top_mask(risk_v2["any"], month)
    rule_top = top_mask(rule_rank, month)
    results["precision_diff_ci"] = boot_precision_diff(y, model_top, rule_top)
    results["true_precision_diff_ci"] = boot_precision_diff(true60, model_top, rule_top)
    closed = test.close_days.to_numpy()
    results["lead_days_median"] = float(np.nanmedian(closed[model_top])) if np.isfinite(closed[model_top]).any() else None
    results["lead_closures_in_top"] = int(np.isfinite(closed[model_top]).sum())
    results["label_noise"] = {"quiet_positive": int(test.y_quiet.sum()),
                              "quiet_no_true_churn_share": float((1 - test.true_churn60[test.y_quiet == 1].astype(float)).mean()),
                              "quiet_hibernator_share": float(test.hibernator[test.y_quiet == 1].astype(float).mean()),
                              "adj_removed": int((test.y_quiet - test.y_quiet_adj).sum()),
                              "true_churn_missed_by_label_share": float((test.y_any[test.true_churn60 == 1] == 0).mean()),
                              "hidden_forced_in_positive_share": float(test.forced_hidden[test.y_any == 1].astype(float).mean())}

    # Политики при равной глубине: одни и те же фильтры, разный порядок клиентов.
    ranks = {"rules": rule_rank, "risk": risk_v2["any"], "risk_adj": risk_s["any"], "risk_x_revenue": risk_s["any"] * revenue,
             "random": np.random.default_rng(s.seed + 1).random(len(test))}
    policies = {}
    for depth in [.01, .05, .1]:
        row = {}
        for name, score in ranks.items():
            usable = np.where(offer_rule > 0, score, -np.inf)
            mask = top_mask(usable, month, depth) & (offer_rule > 0)
            row[name] = policy_value(mask, offer_rule, cost_rule, revenue, keep365, keep60)
        usable = np.where(offer_up > 0, predicted, -np.inf)
        mask = top_mask(usable, month, depth) & (offer_up > 0)
        row["uplift_x_revenue"] = policy_value(mask, offer_up, cost_up, revenue, keep365, keep60)
        oracle = true_gain * .5 - np.stack([q[a]["cost"] for a in [1, 2, 3]], axis=1)
        oracle = np.where(np.stack([q[a]["allowed"] for a in [1, 2, 3]], axis=1), oracle, -np.inf)
        best = oracle.argmax(axis=1) + 1
        best_value = oracle.max(axis=1)
        offer_o = np.where(best_value > 0, best, 0)
        cost_o = np.select([offer_o == 1, offer_o == 2, offer_o == 3], [q[1]["cost"], q[2]["cost"], q[3]["cost"]], 0).astype(float)
        mask = top_mask(np.where(offer_o > 0, best_value, -np.inf), month, depth) & (offer_o > 0)
        row["oracle_bound"] = policy_value(mask, offer_o, cost_o, revenue, keep365, keep60)
        policies[str(depth)] = row
    results["policies"] = policies
    alpha_rows = {}
    for depth in [.01, .05, .1]:
        alpha_rows[str(depth)] = {}
        for name in ["risk_adj", "risk_x_revenue"]:
            usable = np.where(offer_rule > 0, ranks[name], -np.inf)
            mask = top_mask(usable, month, depth) & (offer_rule > 0)
            alpha_rows[str(depth)][name] = policy_value(mask, offer_rule, cost_rule, revenue, keep365, keep60, alpha=.3)
        mask = top_mask(np.where(offer_up > 0, predicted, -np.inf), month, depth) & (offer_up > 0)
        alpha_rows[str(depth)]["uplift_x_revenue"] = policy_value(mask, offer_up, cost_up, revenue, keep365, keep60, alpha=.3)
    results["alpha_03"] = alpha_rows
    true_u = np.max(keep365[:, 1:].astype(float) - keep365[:, [0]].astype(float), axis=1)
    results["uplift_quality"] = {str(a): {"pred_true_corr_mean_by_bin": None} for a in [1, 2, 3]}
    bins = pd.qcut(uplift.max(axis=1), 20, labels=False, duplicates="drop")
    observed = [(keep365[bins == b, 1:].astype(float).mean() - keep365[bins == b, 0].astype(float).mean()) for b in np.unique(bins)]
    predicted_bin = [uplift[bins == b].max(axis=1).mean() for b in np.unique(bins)]
    results["uplift_quality"] = {"bin_pred_true_corr": float(np.corrcoef(predicted_bin, observed)[0, 1]),
                                 "mean_pred_pp": float(uplift.mean() * 100), "true_avg_effect_pp": float((keep365[:, 1:].astype(float).mean() - keep365[:, 0].astype(float).mean()) * 100),
                                 "share_negative_pred": float((uplift.max(axis=1) <= 0).mean())}

    if full:
        from sklift.metrics import qini_auc_score
        qini = {}
        for a in [1, 2, 3]:
            rows = test.arm.isin([0, a]).to_numpy()
            qini[str(a)] = {"n": int(rows.sum()),
                            "qini_uplift": float(qini_auc_score(test.retained365[rows], uplift[rows, a - 1], (test.arm[rows] == a).astype(int))),
                            "qini_risk": float(qini_auc_score(test.retained365[rows], risk_s["any"][rows], (test.arm[rows] == a).astype(int)))}
        results["qini"] = qini

        chosen_v2 = top_mask(risk_v2["any"], month)
        chosen_s = top_mask(risk_s["any"], month)
        fairness = {}
        for label, chosen, target in [("v2_official", chosen_v2, y), ("v2s_official", chosen_s, y), ("v2s_true", chosen_s, true60), ("v2_true", chosen_v2, true60)]:
            fairness[label] = {}
            for col in ["industry", "region"]:
                groups, gap = group_recall(target, chosen, test[col].to_numpy())
                fairness[label][col] = {"groups": groups, "gap": gap, "tests": gap_tests(target, chosen, test[col].to_numpy())}
        results["fairness"] = fairness
        results["seasonal"] = {"v2": seasonal_checks(v2, test, risk_v2["any"], chosen_v2),
                               "v2s": seasonal_checks(v2s, test, risk_s["any"], chosen_s),
                               "g3_features": seasonal_checks(g3, test, risk_g3["any"], top_mask(risk_g3["any"], month))}
        slices = {"low_turnover_q1": test.yoy_ratio.isna(),  # заполняется ниже
                  "short_tenure_lt12": test.tenure < 12, "no_services": test.services == 0,
                  "hibernators": test.hibernator, "forced_hidden": test.forced_hidden,
                  "tourism": test.industry == "tourism", "construction": test.industry == "construction",
                  "professional": test.tariff == "professional", "advanced": test.tariff == "advanced",
                  "complaints_0": test.complaints == 0}
        slices["low_revenue_q1"] = test.revenue_year < test.revenue_year.quantile(.25)
        del slices["low_turnover_q1"]
        table = {}
        for name, rows in slices.items():
            rows = rows.to_numpy()
            target = true60[rows]
            table[name] = {"n": int(rows.sum()), "true_churn_pct": float(target.mean() * 100),
                           "precision_true_pct": float(target[chosen_s[rows]].mean() * 100) if chosen_s[rows].any() else None,
                           "recall_true_pct": float(target[chosen_s[rows]].sum() / max(target.sum(), 1) * 100),
                           "brier": float(((risk_s["any"][rows] - target) ** 2).mean()),
                           "selected_share_pct": float(chosen_s[rows].mean() * 100)}
        results["slices"] = table
        by_month = {}
        for m in np.unique(month):
            rows = month == m
            by_month[str(test.cutoff[rows].iloc[0].date())] = summary(true60[rows], risk_s["any"][rows], month[rows], risk_s["any"][rows])
        results["by_month_true"] = by_month
        results["drift_psi_cal_to_test"] = {c: psi(cal[c], test[c]) for c in FEATURES_V2 if c not in ("industry", "tariff")}
        results["calibration_true"] = []
        order = np.argsort(risk_s["any"])
        for chunk in np.array_split(order, 10):
            results["calibration_true"].append({"predicted": float(risk_s["any"][chunk].mean()), "observed_target": float(test.y_any_adj.to_numpy()[chunk].mean()), "n": int(len(chunk))})
        results["coverage"] = {"class_balance": {"y_close": float(test.y_close.mean()), "y_quiet": float(test.y_quiet.mean()), "y_any": float(y.mean()), "true60": float(true60.mean())},
                               "industry_share": test.industry.value_counts(normalize=True).to_dict(), "tariff_share": test.tariff.value_counts(normalize=True).to_dict(),
                               "reason_share_true_churn365": test.reason[test.true_churn365].value_counts(normalize=True).to_dict(),
                               "missing_share": float(test[FEATURES_V2].isna().mean().max())}
        results["power"] = {f"{p0}_{e}": {"total_90_10": power_total(p0, e), "total_50_50": power_total(p0, e, .5)} for p0, e in [(.75, .10), (.9, .05), (.9, .03), (.925, .06), (.925, .03)]}
        importance = v2s.models["quiet"].get_feature_importance()
        results["importance_quiet"] = dict(sorted(zip(FEATURES_V2, map(float, importance)), key=lambda x: -x[1]))
        results["future_mutation_invariant"] = bool(np.array_equal(
            v2.predict(test.head(200).assign(y_close=1 - test.y_close.head(200), y_quiet=1 - test.y_quiet.head(200), t0=0, retained365=0))["any"], risk_v2["any"][:200]))
        results["split_cuts_valid"] = bool(split_cuts_are_valid())
        MODELS.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
        with (MODELS / "bundle.pkl").open("wb") as f:
            pickle.dump({"v2": v2, "v2s": v2s, "g3": g3, "rules": rules, "uplift": uplift_model, "profile": profile}, f)
        keep = test.copy()
        keep["risk_v2"] = risk_v2["any"]; keep["risk_s"] = risk_s["any"]; keep["risk_rule"] = p_rule
        for a in [1, 2, 3]:
            keep[f"uplift_{a}"] = uplift[:, a - 1]
        keep.to_pickle(DATA / "test.pkl")
        cal[FEATURES_V2].sample(min(50_000, len(cal)), random_state=1).to_pickle(DATA / "reference.pkl")
    results["elapsed_seconds"] = time.time() - started
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "dev":
        out = run(DEV)
        dump("_dev.json", out)
        print(json.dumps({k: out[k] for k in ["funnel", "label_noise", "lead_days_median", "elapsed_seconds"]}, ensure_ascii=False, indent=1))
        for name, row in out["risk"]["official_target"].items():
            print(name, {k: round(v, 4) for k, v in row.items()})
    if mode in ("all", "main"):
        out = run(S0)
        dump("main.json", out)
        print("main done", out["elapsed_seconds"])
    if mode in ("all", "sens"):
        sens = {}
        for s in SENSITIVITY:
            r = run(s, full=False)
            sens[s.name] = {"scenario": r["scenario"], "risk": r["risk"]["official_target"], "true": r["risk"]["true_churn60"],
                            "policies": r["policies"], "alpha_03": r["alpha_03"], "uplift_quality": r["uplift_quality"], "label_noise": r["label_noise"]}
            dump("sensitivity.json", sens)
            print(s.name, "done", r["elapsed_seconds"])
