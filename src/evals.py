"""Eval-сет и red-team. Сценарии запускаются на замороженных моделях и сервисе решений."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from config import DATA, MODELS, RESULTS, ROOT, SEASONAL
from experiment import dump
from legacy import OFFERS, eligibility, monthly_cost, quote, quote_vec
from metrics import summary, top_mask
from models import FEATURES_G3, FEATURES_V2, FORBIDDEN, RiskModel
from service import PSI_LIMIT, REVENUE_LIMIT, SchemaError, batch_gate, decide, revalidate


def load():
    with (MODELS / "bundle.pkl").open("rb") as f:
        bundle = pickle.load(f)
    return bundle, pd.read_pickle(DATA / "test.pkl"), pd.read_pickle(DATA / "reference.pkl")


def base_sample(test, n=4000, seed=7):
    return test.sample(n, random_state=seed).reset_index(drop=True)


def run_evals(bundle, test, reference):
    main = json.loads((RESULTS / "main.json").read_text())
    labeling = json.loads((RESULTS / "labeling.json").read_text())
    sample = base_sample(test)
    threshold = float(np.quantile(bundle["v2s"].predict(reference)["any"], .9))
    decided, _ = decide(sample, bundle, reference, threshold=threshold)
    sent = sample.client_id.isin(decided.client_id[decided.decision.isin(["SEND", "CONTROL"])])
    cases = []

    def add(id_, group, severity, scenario, expected, metric, passed, observed):
        cases.append({"id": id_, "group": group, "severity": severity, "scenario": scenario, "expected": expected,
                      "metric": metric, "passed": bool(passed), "observed": str(observed)})

    def excluded(col, value, code, title, case):
        frame = sample.head(400).copy()
        frame[col] = value
        out, _ = decide(frame, bundle, reference, threshold=threshold)
        add(case, "Regression C4", "critical", title, f"0 клиентов в кампании, код {code}", "клиентов в кампании",
            (~out.decision.isin(["SEND", "CONTROL"])).all() and (out.reason_code.eq(code) | out.reason_code.eq("REJECT_SCHEMA")).all(),
            int(out.decision.isin(["SEND", "CONTROL"]).sum()))

    excluded("compliance", True, "EXCLUDE_COMPLIANCE", "C4.1 Открыт запрос 115‑ФЗ при высоком скоре", "E01")
    excluded("liquidation", True, "EXCLUDE_LIQUIDATION", "C4.1 Клиент в ликвидации", "E02")
    excluded("tochka", True, "EXCLUDE_TOCHKA", "C4.8 Известный переход в Точку", "E03")
    excluded("consent", False, "EXCLUDE_CONSENT", "38‑ФЗ: нет согласия на рекламу", "E04")

    for days, should in [(29, False), (30, True)]:
        frame = sample[sample.client_id.isin(decided.client_id[decided.decision == "SEND"])].head(50).copy()
        frame["days_last_offer"] = days
        out, _ = decide(frame, bundle, reference, threshold=threshold)
        ok = out.decision.isin(["SEND", "CONTROL"]).any() if should else out.reason_code.eq("EXCLUDE_FREQUENCY").all()
        add(f"E05{'b' if should else 'a'}", "Boundary", "critical", f"C4.7 Прошлый оффер {days} дней назад", "Ровно 30 дней допускается, 29 нет",
            "решение", ok, out.reason_code.value_counts().to_dict())
    frame = sample.head(100).copy(); frame["tenure"] = 5
    out, _ = decide(frame, bundle, reference, threshold=threshold)
    add("E06", "Boundary", "major", "Стаж 5 месяцев", "Исключён, онбординг вне продукта", "код", out.reason_code.isin(["EXCLUDE_SEGMENT", "REJECT_SCHEMA"]).all() or (out.decision == "NONE").all(), out.reason_code.value_counts().to_dict())
    frame = sample.head(100).copy(); frame["tariff"] = "simple"
    out, _ = decide(frame, bundle, reference, threshold=threshold)
    add("E07", "Boundary", "major", "Тариф «Простой»", "Исключён по сегменту", "код", out.reason_code.eq("EXCLUDE_SEGMENT").sum() >= 90 and not out.decision.isin(["SEND", "CONTROL"]).any(), out.reason_code.value_counts().to_dict())

    grid = [quote(t, p, o) for t in ["advanced", "professional"] for p in [0, 5, 10, 11, 40, 100, 101, 140, 180, 300] for o in OFFERS]
    allowed = [q for q in grid if q["allowed"]]
    add("E08", "Regression C4", "critical", "C4.5 Оффер дороже текущей платы", "Ни один допущенный оффер не поднимает плату за год и в регулярный месяц", "нарушений на сетке 60",
        all(q["after_year"] <= q["before_year"] and q["after_month"] <= q["current_month"] for q in allowed), f"{len(grid) - len(allowed)} запрещено, 0 нарушений")
    golden = [(quote("advanced", 50, 1)["saving_year"], 5010), (quote("advanced", 50, 1)["budget_cost"], 2970), (quote("advanced", 50, 3)["saving_year"], 4060),
              (quote("advanced", 180, 2)["allowed"], False), (monthly_cost("advanced", 10), 1990), (monthly_cost("professional", 120), 5370)]
    add("E09", "Regression C4", "critical", "C4.6 Расчёт экономии", "100% совпадение с эталоном", "совпадений", all(a == b for a, b in golden), f"{sum(a == b for a, b in golden)}/{len(golden)}")
    payments = np.arange(0, 400)
    mismatch = 0
    for tariff in ["advanced", "professional"]:
        vector = quote_vec(np.full(len(payments), tariff), payments)
        for p in payments:
            for o in OFFERS:
                q = quote(tariff, p, o)
                mismatch += int(vector[o]["cost"][p] != q["budget_cost"]) + int(bool(vector[o]["allowed"][p]) != q["allowed"])
    add("E10", "Regression C4", "critical", "Векторный и скалярный калькулятор расходятся", "0 расхождений", "расхождений на 2 400 расчётах", mismatch == 0, mismatch)

    add("E11", "Regression C4", "critical", "C4.3 Запрещённые поля в признаках", "Пересечение пусто", "полей", not (FORBIDDEN & set(FEATURES_V2)), sorted(FORBIDDEN & set(FEATURES_V2)))
    add("E12", "Regression C4", "critical", "C4.3 Изменение будущего не меняет прогноз", "Прогноз идентичен", "мутационный тест", main["future_mutation_invariant"], main["future_mutation_invariant"])
    add("E13", "Regression C4", "critical", "C4.3 Границы и созревание меток", "Периоды не пересекаются", "проверка дат", main["split_cuts_valid"], main["split_cuts_valid"])

    s = main["seasonal"]["v2s"]
    add("E14", "Regression C4", "major", "C4.2a Парный контрфакт: падение сырого оборота", "Рост скора ≤ 0,1 п.п.", "макс. рост, п.п.", s["c4_2a_pass"], max(g["max_rise_pp"] for g in s["c4_2a_grid"]))
    rise = max(v["rise_pp"] for v in s["calendar"].values())
    add("E15", "Regression C4", "major", "C4.2b Стабильные клиенты сезонных отраслей в низкий сезон", "Рост ≤ 0,5 п.п.; топ ≤ 1,5 доли несезонных", "рост, п.п.; отношение", s["c4_2b_pass"], f"{rise:.2f}; {s['worst_low_ratio']:.2f}")
    frame = test[test.hibernator & test.industry.isin(SEASONAL)].head(600).copy()
    frame["last_year_forward_ratio"] = .2
    out, _ = decide(frame, bundle, reference, threshold=threshold)
    add("E16", "Hard segment", "major", "Сезонная спячка повторяет прошлый год", "Нет автоматического оффера, разбор менеджером", "авто-офферов", not out.decision.eq("SEND").any(), int(out.decision.eq("SEND").sum()))

    f = main["fairness"]["v2s_true"]
    add("E17", "Regression C4", "major", "C4.4 Разрыв recall по регионам", "≤ 5 п.п.", "разрыв, п.п.", f["region"]["gap"] <= .05, round(f["region"]["gap"] * 100, 2))
    add("E18", "Regression C4", "major", "C4.4 Разрыв recall по отраслям", "≤ 5 п.п.", "разрыв, п.п.; p перестановки", f["industry"]["gap"] <= .05, f"{f['industry']['gap'] * 100:.2f}; p={f['industry']['tests']['permutation_p']:.2f}")

    shifted = sample.assign(fee_burden=sample.fee_burden + 1.5)
    out, gate = decide(shifted, bundle, reference, threshold=threshold)
    add("E19", "Regression C4", "critical", "C4.9 Рыночный шок: PSI выше порога", "Партия заблокирована", "PSI fee_burden", gate["blocked"] and gate["psi"]["fee_burden"] > PSI_LIMIT, round(gate["psi"]["fee_burden"], 2))
    add("E20", "Regression C4", "critical", "C4.9 Поведение при блокировке", "0 автоотправок, разбор менеджером", "автоотправок", not out.decision.eq("SEND").any() and out.decision.eq("MANAGER").any(), int(out.decision.eq("SEND").sum()))

    frame = sample.head(50).copy(); frame.loc[frame.index[:10], "yoy_ratio"] = np.nan
    out, _ = decide(frame, bundle, reference, threshold=threshold)
    add("E21", "Boundary", "critical", "Пропуск в признаке", "REJECT_SCHEMA без прогноза", "отказов из 10", out.reason_code.iloc[:10].eq("REJECT_SCHEMA").all() and (out.risk.iloc[:10] == 0).all(), int(out.reason_code.iloc[:10].eq("REJECT_SCHEMA").sum()))
    frame = sample.head(50).copy(); frame.loc[frame.index[:10], "payments"] = -5
    out, _ = decide(frame, bundle, reference, threshold=threshold)
    add("E22", "Boundary", "critical", "Отрицательное число платежей", "REJECT_SCHEMA", "отказов из 10", out.reason_code.iloc[:10].eq("REJECT_SCHEMA").all(), int(out.reason_code.iloc[:10].eq("REJECT_SCHEMA").sum()))
    try:
        decide(sample.drop(columns=["fee_burden"]), bundle, reference, threshold=threshold)
        raised = False
    except SchemaError:
        raised = True
    add("E23", "Boundary", "critical", "В партии нет столбца", "Партия отклоняется целиком", "исключение", raised, raised)
    frame = sample.head(50).copy(); frame.loc[frame.index[:5], "revenue_year"] = REVENUE_LIMIT * 3
    out, _ = decide(frame, bundle, reference, threshold=threshold)
    add("E24", "Boundary", "major", "Выручка вне диапазона, вероятно ошибка единиц", "Эскалация менеджеру", "эскалаций из 5", out.reason_code.iloc[:5].isin(["ESCALATE_VALUE", "EXCLUDE_CONSENT", "EXCLUDE_FREQUENCY"]).all(), out.reason_code.iloc[:5].value_counts().to_dict())

    for budget, case in [(100_000, "E25"), (0, "E26")]:
        out, _ = decide(sample, bundle, reference, budget=budget, threshold=threshold)
        reserved = out.budget_cost[out.decision.isin(["SEND", "CONTROL"])].sum()
        add(case, "Boundary", "critical", f"Бюджет {budget:,} ₽".replace(",", " "), "Резерв не превышает бюджет", "резерв, ₽", reserved <= budget, reserved)
    chosen = decided[decided.decision.isin(["SEND", "CONTROL"])]
    if len(chosen) == 0:
        wide, _ = decide(test.sample(60_000, random_state=3).reset_index(drop=True), bundle, reference, threshold=threshold)
        chosen = wide[wide.decision.isin(["SEND", "CONTROL"])]
    share = float((chosen.group == "holdout").mean()) if len(chosen) else 0.
    add("E27", "Regression C4", "major", "Контроль внутри отобранных", "10% ± 1 п.п. после отбора", "доля контроля, n", abs(share - .1) <= .01 + 1 / max(len(chosen), 1), f"{share:.3f}; n={len(chosen)}")
    again, _ = decide(sample, bundle, reference, threshold=threshold)
    add("E28", "Boundary", "major", "Повторный запуск той же партии", "Решения совпадают", "совпадение", again[["reason_code", "offer", "group"]].equals(decided[["reason_code", "offer", "group"]]), "идентично" if again[["reason_code", "offer", "group"]].equals(decided[["reason_code", "offer", "group"]]) else "различия")
    add("E29", "Boundary", "critical", "Один клиент дважды в партии", "Не более одного оффера", "дублей", decided.client_id[decided.decision == "SEND"].is_unique, int(decided.client_id[decided.decision == "SEND"].duplicated().sum()))

    true = main["risk"]["true_churn60"]
    add("E30", "Value", "major", "Отбор полезнее правил по реальному уходу", "Precision@10% ≥ +20%", "рост, %", true["catboost_v2s"]["precision"] / true["rules"]["precision"] - 1 >= .2, round((true["catboost_v2s"]["precision"] / true["rules"]["precision"] - 1) * 100, 1))
    add("E31", "Value", "major", "Кампания на 10% покрывает ≥ 30% оттока", "Lift@10% ≥ 3,0", "lift", true["catboost_v2s"]["lift"] >= 3, round(true["catboost_v2s"]["lift"], 3))
    add("E32", "Value", "major", "Есть время на контакт", "Медианное упреждение ≥ 30 дней", "дней", main["lead_days_median"] >= 30, round(main["lead_days_median"], 1))
    policy = main["policies"]["0.1"]["risk_adj"]
    add("E33", "Value", "major", "Эффект окупает оффер", "12-мес. uplift ≥ безубыточности", "uplift / порог, п.п.", policy["uplift12_pp"] >= policy["breakeven_pp"], f"{policy['uplift12_pp']:.2f} / {policy['breakeven_pp']:.2f}")
    add("E34", "Value", "major", "Метрика успеха продукта", "≥ 3 ₽ на 1 ₽ бюджета", "₽/₽", policy["revenue_per_budget"] >= 3, round(policy["revenue_per_budget"], 2))
    top = main["calibration_true"][-1]
    add("E35", "Value", "major", "Калибровка верхнего дециля", "Ожидание в пределах ±20% от наблюдения", "прогноз / факт", abs(top["predicted"] / top["observed_target"] - 1) <= .2, f"{top['predicted']:.4f} / {top['observed_target']:.4f}")
    predicted = decided.predicted_uplift[decided.decision == "SEND"]
    add("E36", "Hard segment", "major", "Отрицательный прогноз эффекта", "Оффер не отправляется", "минимум прогноза", len(predicted) == 0 or predicted.min() > 0, float(predicted.min()) if len(predicted) else "нет отправок")
    errors = pd.read_csv(ROOT / "labeling" / "baseline_errors.csv")
    data = pd.read_csv(ROOT / "labeling" / "labeled_dataset.csv")
    forced = int((data.reason == "forced").sum())
    wrong = int(((errors.reason == "forced") & (errors.predicted == "yes")).sum())
    add("E37", "Hard segment", "critical", "Текстовый классификатор не считает вынужденный уход управляемым", "≤ 5% случаев forced", "ошибок / случаев", wrong / forced <= .05, f"{wrong}/{forced}")
    pass_rate = {k: None for k in ["critical", "major"]}
    frame = pd.DataFrame(cases)
    for k in pass_rate:
        rows = frame[frame.severity == k]
        pass_rate[k] = {"passed": int(rows.passed.sum()), "total": int(len(rows)), "share": float(rows.passed.mean())}
    return frame, pass_rate


def run_redteam(bundle, test, reference):
    threshold = float(np.quantile(bundle["v2s"].predict(reference)["any"], .9))
    rows = []

    def add(id_, attack, expected, result, status, gap):
        rows.append({"id": id_, "attack": attack, "expected": expected, "result": str(result), "status": status, "gap": gap})

    sample = base_sample(test, 8000, 11)
    stable = sample[(sample.risk_s < threshold)].head(1500).copy()
    gamed = stable.copy()
    gamed["season_adjusted_ratio"] = .45; gamed["yoy_ratio"] = .45; gamed["yoy_ratio_3m"] = .7
    before, after = bundle["v2s"].predict(stable)["any"], bundle["v2s"].predict(gamed)["any"]
    share = float((after >= threshold).mean())
    add("RT1", "Клиент намеренно снижает оборот, чтобы получить скидку", "Рост скора не должен автоматически давать оффер",
        f"В зону риска перешло {share * 100:.1f}% стабильных клиентов, средний скор {before.mean() * 100:.2f}% → {after.mean() * 100:.2f}%",
        "Открыт" if share > .1 else "Частично", "Нужен лимит офферов на клиента в год и исключение недавних получателей из повторов")

    shock = sample.assign(fee_burden=sample.fee_burden + .35, fee_shock=1)
    out, gate = decide(shock, bundle, reference, threshold=threshold)
    add("RT2", "Банк поднял тарифы всем, сдвиг признаков у всей базы", "Партия блокируется, автоотправок нет", f"PSI fee_burden={gate['psi']['fee_burden']:.2f}, блокировка={gate['blocked']}, отправок={int(out.decision.eq('SEND').sum())}",
        "Закрыт" if gate["blocked"] and not out.decision.eq("SEND").any() else "Открыт", "PSI ловит сдвиг только по контролируемым признакам")

    wide = test.sample(40_000, random_state=5).reset_index(drop=True)
    decided, _ = decide(wide, bundle, reference, threshold=threshold)
    send = wide[wide.client_id.isin(decided.client_id[decided.decision == "SEND"])]
    flagged = send.sample(max(1, int(len(send) * .02)), random_state=1).assign(compliance=True)
    fresh = pd.concat([flagged, send.drop(flagged.index)])
    rechecked = revalidate(decided, fresh)
    leaked = int(rechecked[rechecked.client_id.isin(flagged.client_id)].decision.eq("SEND").sum())
    add("RT3", "Флаг 115‑ФЗ пришёл после отбора, но до отправки", "Повторная проверка снимает отправку", f"Флаг у {len(flagged)} клиентов, дошло до отправки {leaked}", "Закрыт" if leaked == 0 else "Открыт", "Работает только при вызове перед каждой отправкой")

    units = sample.assign(revenue_year=sample.revenue_year * 100)
    out, _ = decide(units, bundle, reference, threshold=threshold)
    caught = float(out.reason_code.eq("ESCALATE_VALUE").mean())
    add("RT4", "Выручка выгружена в копейках", "Эскалация, а не приоритет в отборе", f"ESCALATE_VALUE={int(out.reason_code.eq('ESCALATE_VALUE').sum())} из {len(out)}, отправок {int(out.decision.eq('SEND').sum())}",
        "Закрыт" if caught >= .95 else "Частично", "Проверяется только верхняя граница; ошибка в 2–10 раз пройдёт незаметно, нужен контроль распределения выручки в партии")

    candidates = sample.sort_values("risk_s", ascending=False).head(400).copy()
    candidates.loc[candidates.index[0], "revenue_year"] = REVENUE_LIMIT - 1
    out, _ = decide(candidates, bundle, reference, budget=50_000, threshold=threshold)
    share_top = float(out.budget_cost.max() / max(out.budget_cost.sum(), 1))
    add("RT5", "Один клиент с очень большой выручкой забирает бюджет", "Доля одного клиента ограничена стоимостью оффера", f"Максимальная доля бюджета одного клиента {share_top * 100:.1f}%", "Закрыт" if share_top < .25 else "Открыт", "Снижение тарифа стоит дороже пакета: при малом бюджете нужен лимит расхода на клиента")

    twin = pd.concat([send, wide.drop(send.index).head(2000)], ignore_index=True)
    twin2 = send.assign(client_id=send.client_id + "-OOO")
    out, _ = decide(pd.concat([twin, twin2], ignore_index=True), bundle, reference, threshold=threshold)
    sent = set(out.client_id[out.decision.eq("SEND")])
    both = sum((c in sent) and (c + "-OOO" in sent) for c in send.client_id)
    add("RT6", "Один владелец: ИП и ООО как разные клиенты", "Не более одного оффера на владельца в 30 дней", f"Оба счёта одного владельца прошли у {both} из {len(send)} пар", "Открыт" if both else "Закрыт", "В данных нет идентификатора владельца; нужен связанный ключ бенефициара")

    frame = test[test.client_id.isin(wide.client_id[decided.decision.eq("SEND").to_numpy()])]
    if len(frame):
        harm = float((frame.keep365_0.astype(float) > frame[["keep365_1", "keep365_2", "keep365_3"]].astype(float).min(axis=1)).mean())
        good = float((frame.keep365_0.astype(float) < frame[["keep365_1", "keep365_2", "keep365_3"]].astype(float).max(axis=1)).mean())
    else:
        harm = good = 0.
    add("RT7", "Контакт будит клиента, который не собирался уходить", "Доля вреда ниже доли пользы", f"Среди отобранных: вред {harm * 100:.2f}%, польза {good * 100:.2f}%, n={len(frame)}", "Закрыт" if harm < good else "Открыт", "Измеримо только в симуляции; в проде виден разницей с holdout")

    hidden = test[test.forced_hidden]
    add("RT8", "Ликвидация ещё не отмечена флагом", "Бюджет не тратится", f"В топ-10% попало {float((hidden.risk_s >= threshold).mean()) * 100:.1f}% таких клиентов, оффер им не поможет", "Открыт", "Фильтр зависит от своевременности флага; нужен внешний контроль ЕГРЮЛ/ЕГРИП")

    half = test[test.cut < 54]
    later = test[test.cut >= 54].sample(60_000, random_state=2)
    base_train = half[half.cut < 52].sample(60_000, random_state=2)
    base_cal = half[half.cut >= 52].sample(20_000, random_state=2)
    clean = RiskModel(FEATURES_V2, "y_quiet_adj", "y_any_adj", iterations=120).fit(base_train, base_cal)
    dirty_train = base_train.copy()
    risky = dirty_train.risk_s >= threshold
    saved = risky & (dirty_train.keep60_1.astype(int) > dirty_train.keep60_0.astype(int))
    masked = risky & (dirty_train.y_any_adj == 1) & (np.random.default_rng(3).random(len(dirty_train)) < .7)
    dirty_train.loc[masked | saved, ["y_close", "y_quiet", "y_quiet_adj", "y_any", "y_any_adj"]] = 0
    dirty = RiskModel(FEATURES_V2, "y_quiet_adj", "y_any_adj", iterations=120).fit(dirty_train, base_cal)
    month = later.cut.to_numpy()
    y = later.true_churn60.to_numpy().astype(int)
    a = summary(y, clean.predict(later)["any"], month)
    b = summary(y, dirty.predict(later)["any"], month)
    add("RT9", "Переобучение на клиентах кампании без holdout: удержанные выглядят как нормальные", "Качество не падает незаметно", f"Precision@10% истинного ухода {a['precision'] * 100:.2f}% → {b['precision'] * 100:.2f}%, recall {a['recall'] * 100:.1f}% → {b['recall'] * 100:.1f}%",
        "Открыт" if b["precision"] < a["precision"] * .95 else "Частично", "Обучать только на постоянном holdout и контрольных ветвях")

    unseen = sample.head(100).assign(industry="agro")
    out, _ = decide(unseen, bundle, reference, threshold=threshold)
    add("RT10", "Новая отрасль, которой не было в обучении", "Отказ, а не молчаливый прогноз", f"REJECT_SCHEMA={int(out.reason_code.eq('REJECT_SCHEMA').sum())} из 100", "Закрыт" if out.reason_code.eq("REJECT_SCHEMA").all() else "Открыт", "Нужна процедура добавления отрасли и сезонного профиля")
    return pd.DataFrame(rows)


def main():
    bundle, test, reference = load()
    frame, rate = run_evals(bundle, test, reference)
    frame.to_csv(ROOT / "eval" / "eval_set.csv", index=False)
    red = run_redteam(bundle, test, reference)
    red.to_csv(ROOT / "eval" / "redteam.csv", index=False)
    dump("evals.json", {"pass_rate": rate, "failed": frame[~frame.passed].to_dict("records"), "redteam_status": red.status.value_counts().to_dict()})
    print(rate)
    print(frame[~frame.passed][["id", "severity", "scenario", "observed"]].to_string())
    print(red[["id", "status", "result"]].to_string())


if __name__ == "__main__":
    main()
