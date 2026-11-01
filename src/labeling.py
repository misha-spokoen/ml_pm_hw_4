"""Разметка: развёртка меток, выборка второго прохода, каппа, бейзлайн на размеченных данных."""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from config import RESULTS, ROOT
from metrics import wilson

LAB = ROOT / "labeling"
MAPS = {"outcome": {"L": "left", "S": "stayed", "U": "unknown"},
        "reason": {"fee": "price_fee", "fix": "price_fixed", "frc": "forced", "crd": "credit_counterparty",
                   "svc": "service", "oth": "other", "unk": "unknown"},
        "manageable": {"y": "yes", "n": "no", "a": "na"},
        "action": {"p": "package", "l": "lower_tariff", "y": "annual", "n": "none"}}
KEYWORDS = r"комисс|тариф|абонент|плат[аеу]\b|плату|дорог|дешев|дешёв|цена|цены|стоимост|обслуживани"
EXCLUDE = r"ликвид|закрыл ип|блокир|115|кредит|лизинг|заказчик"


def expand(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append({"id": f"GEN-{parts[0]}", "outcome": MAPS["outcome"][parts[1]], "reason": MAPS["reason"][parts[2]],
                     "manageable": MAPS["manageable"][parts[3]],
                     "action": MAPS["action"][parts[4]] if len(parts) > 4 and parts[4] in MAPS["action"] else "none",
                     "hard": int(parts[-1] == "h")})
    return pd.DataFrame(rows)


def select_round2(n=70, seed=20261102):
    """Стратификация только по первому исходу; порядок перемешан, метки не показываются."""
    cases = pd.read_csv(LAB / "cases_generated.csv")
    first = expand(LAB / "labels_round1_compact.txt")
    frame = cases.merge(first[["id", "outcome"]], on="id")
    picked = frame.groupby("outcome", group_keys=False).sample(frac=n / len(frame), random_state=seed)
    return picked.sample(frac=1, random_state=seed)[["id", "text"]]


def kappa_ci(a, b, reps=2000, seed=1):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    point = cohen_kappa_score(a, b)
    boots = []
    for _ in range(reps):
        s = rng.integers(0, len(a), len(a))
        if len(set(a[s]) | set(b[s])) > 1:
            boots.append(cohen_kappa_score(a[s], b[s]))
    return {"kappa": float(point), "agreement": float((a == b).mean()), "n": int(len(a)),
            "ci95": [float(x) for x in np.nanquantile(boots, [.025, .975])]}


def agreement(first, second):
    merged = first.merge(second, on="id", suffixes=("_1", "_2"))
    out = {c: kappa_ci(merged[f"{c}_1"], merged[f"{c}_2"]) for c in ["outcome", "reason", "manageable", "action"]}
    disagreements = merged[(merged.outcome_1 != merged.outcome_2) | (merged.reason_1 != merged.reason_2)
                           | (merged.manageable_1 != merged.manageable_2) | (merged.action_1 != merged.action_2)]
    return out, merged, disagreements


def heuristic(texts):
    """Эвристика на ключевых словах: цена названа, а вынужденных и кредитных маркеров нет."""
    lower = pd.Series(texts).str.lower()
    return (lower.str.contains(KEYWORDS, regex=True) & ~lower.str.contains(EXCLUDE, regex=True)).astype(int).to_numpy()


def score(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def boot_delta(y, a, b, reps=2000, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        s = rng.integers(0, len(y), len(y))
        out.append(score(y[s], a[s])["f1"] - score(y[s], b[s])["f1"])
    return [float(x) for x in np.quantile(out, [.025, .975])]


def model():
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
                         LogisticRegression(C=10, class_weight="balanced", max_iter=2000))


def baseline(data):
    """Цель: управляемая причина среди случаев с названной причиной. Нужна для выбора действия."""
    known = data[data.manageable != "na"].copy()
    y = (known.manageable == "yes").astype(int).to_numpy()
    # Отложенная выборка — позже по дате; в обучение случаи будущего не попадают.
    cut = sorted(known.month)[int(len(known) * .7)]
    train, test = (known.month < cut).to_numpy(), (known.month >= cut).to_numpy()
    fitted = model().fit(known.text[train], y[train])
    proba = fitted.predict_proba(known.text[test])[:, 1]
    pred = (proba >= .5).astype(int)
    keyword = heuristic(known.text[test].tolist())
    holdout = {"split_month": cut, "train": int(train.sum()), "test": int(test.sum()), "positive_test": int(y[test].sum()),
               "model": score(y[test], pred), "keywords": score(y[test], keyword),
               "f1_delta_ci95": boot_delta(y[test], pred, keyword),
               "precision_ci95": wilson(int((pred & y[test]).sum()), int(pred.sum()))}
    out_of_fold = np.zeros(len(known))
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
    for a, b in folds.split(known.text, y, groups=known.platform + known.month):
        out_of_fold[b] = model().fit(known.text.iloc[a], y[a]).predict_proba(known.text.iloc[b])[:, 1]
    cv_pred = (out_of_fold >= .5).astype(int)
    cv = {"model": score(y, cv_pred), "keywords": score(y, heuristic(known.text.tolist()))}
    errors = known.iloc[np.where(cv_pred != y)[0]][["id", "text", "manageable", "reason", "hard"]].copy()
    errors["predicted"] = np.where(cv_pred[cv_pred != y] == 1, "yes", "no")
    return {"holdout": holdout, "cross_validation": cv, "n": int(len(known)), "positive_share": float(y.mean())}, errors


def main():
    cases = pd.read_csv(LAB / "cases_generated.csv")
    first = expand(LAB / "labels_round1_compact.txt")
    second = expand(LAB / "labels_round2_compact.txt")
    agree, merged, disagreements = agreement(first, second)
    data = cases.merge(first, on="id")
    assert len(data) == len(cases) and data.id.is_unique and data.text.is_unique
    data.to_csv(LAB / "labeled_dataset.csv", index=False)
    merged.merge(cases[["id", "text"]], on="id").to_csv(LAB / "double_labeled.csv", index=False)
    disagreements.merge(cases[["id", "text"]], on="id").to_csv(LAB / "disagreements.csv", index=False)
    left = data[data.outcome == "left"]
    known_left = left[left.manageable != "na"]
    k = int((left.manageable == "yes").sum())
    share = {"left": int(len(left)), "manageable": k, "unknown_reason": int((left.manageable == "na").sum()),
             "share_all_left": k / len(left), "ci95": wilson(k, len(left)),
             "share_known_reason": k / len(known_left), "ci95_known": wilson(k, len(known_left))}
    base, errors = baseline(data)
    errors.to_csv(LAB / "baseline_errors.csv", index=False)
    distribution = {c: data[c].value_counts().to_dict() for c in ["outcome", "reason", "manageable", "action"]}
    result = {"n": int(len(data)), "distribution": distribution, "hard_share": float(data.hard.mean()), "agreement": agree,
              "agreement_hard_only": {c: kappa_ci(merged[merged.hard_1 == 1][f"{c}_1"], merged[merged.hard_1 == 1][f"{c}_2"]) for c in ["outcome", "manageable"]}
              if merged.hard_1.sum() > 3 else None,
              "disagreement_count": int(len(disagreements)), "manageable_share": share, "baseline": base}
    (RESULTS / "labeling.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: result[k] for k in ["agreement", "disagreement_count", "manageable_share", "baseline"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "select":
        for row in select_round2().itertuples():
            print(row.id[4:], row.text)
    else:
        main()
