"""Собирает и выполняет baseline.ipynb, чтобы результаты ячеек были видны без запуска."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELLS = [
    md("# Бейзлайн Gate 4–5\n\nДанные учебные и сгенерированы. Ноутбук показывает два независимых замера: модель на размеченном корпусе и модели на панели против правил Gate 3. Их метрики не сравниваются между собой."),
    code("""import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
from labeling import agreement, baseline, expand, heuristic, score, LAB
pd.set_option("display.max_colwidth", 90, "display.width", 140)"""),
    md("## 1. Размеченный корпус\n\n277 уникальных сгенерированных случаев. Размечал Claude по гайдлайну; второй проход — 70 случаев."),
    code("""data = pd.read_csv(LAB / "labeled_dataset.csv")
print(len(data), data.text.is_unique)
for col in ["outcome", "reason", "manageable", "action"]:
    print(data[col].value_counts().to_dict())
data.sample(5, random_state=1)[["id", "text", "outcome", "reason", "manageable", "action"]]"""),
    md("## 2. Согласованность проходов"),
    code("""first, second = expand(LAB / "labels_round1_compact.txt"), expand(LAB / "labels_round2_compact.txt")
agree, merged, disagreements = agreement(first, second)
display(pd.DataFrame(agree).T[["kappa", "agreement", "n", "ci95"]])
disagreements[["id", "outcome_1", "outcome_2", "reason_1", "reason_2", "manageable_1", "manageable_2"]]"""),
    md("## 3. Бейзлайн на размеченных данных\n\nЦель — управляемая причина. Отложенная часть позже по дате."),
    code("""result, errors = baseline(data)
display(pd.DataFrame({k: result["holdout"][k] for k in ["model", "keywords"]}).T)
print({k: result["holdout"][k] for k in ["split_month", "train", "test", "positive_test", "f1_delta_ci95"]})
errors[["text", "manageable", "predicted", "reason"]].head(8)"""),
    md("## 4. Панель: модели против правил Gate 3\n\nТест: 12 месяцев 2025 года. Истинный уход — скрытый эталон симулятора, в обучении не участвует."),
    code("""main = json.loads((ROOT / "results" / "main.json").read_text())
names = ["rules", "logistic_v2", "catboost_g3_features", "catboost_v2", "catboost_v2s"]
for title, key in [("Официальный таргет", "official_target"), ("Истинный уход за 60 дней", "true_churn60")]:
    print(title)
    display(pd.DataFrame(main["risk"][key]).T.loc[names, ["positive", "base_rate", "precision", "lift", "recall", "brier"]].astype(float).round(4))"""),
    md("## 5. Честность замера"),
    code("""print("Границы периодов валидны:", main["split_cuts_valid"])
print("Мутация будущего не меняет прогноз:", main["future_mutation_invariant"])
print("Шум метки:", {k: round(v, 3) for k, v in main["label_noise"].items()})
print("Важность признаков, тихий отток:")
pd.Series(main["importance_quiet"]).round(1).head(10)"""),
    md("## 6. Эффект и экономика\n\nГлубина 10%, отбор моделью v2s, истинный эффект по скрытым потенциальным исходам."),
    code("""rows = pd.DataFrame(main["policies"]["0.1"]).T
rows[["selected", "uplift12_pp", "uplift60_pp", "mean_cost", "breakeven_pp", "revenue_per_budget", "after_ops_per_1000"]].astype(float).round(2)"""),
    md("## 7. Эвалы"),
    code("""evals = pd.read_csv(ROOT / "eval" / "eval_set.csv")
print(evals.groupby("severity").passed.agg(["sum", "count"]))
evals[~evals.passed][["id", "severity", "scenario", "expected", "observed"]]"""),
]

nb = nbf.v4.new_notebook(cells=CELLS)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(nb, ROOT / "baseline.ipynb")
print("notebook executed")
