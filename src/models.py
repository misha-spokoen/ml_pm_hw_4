"""Риск-модели, простой бейзлайн и T-learner на 12-месячном удержании."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import make_column_transformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import SEED
from legacy import rule_score

CATEGORICAL = ["industry", "tariff"]
FEATURES_V2 = ["season_adjusted_ratio", "yoy_ratio", "yoy_ratio_3m", "last_year_forward_ratio", "days_no_incoming",
               "fee_shock", "fee_burden", "self_transfer_share", "complaints", "services", "tenure",
               "industry", "tariff", "payments", "cash_buffer"]
FEATURES_G3 = ["turnover_ratio", "season_adjusted_ratio", "yoy_ratio", "days_no_incoming", "fee_shock", "fee_burden",
               "self_transfer_share", "complaints", "services", "tenure", "industry", "region", "tariff",
               "payments", "cash_buffer"]
FORBIDDEN = {"y_close", "y_quiet", "y_quiet_adj", "y_any", "y_any_adj", "retained60", "retained365", "arm",
             "compliance", "liquidation", "tochka", "consent", "client_id", "split", "cut", "cutoff", "revenue_year"}


def cat_indices(features):
    return [c for c in features if c in ("industry", "tariff", "region")]


class RiskModel:
    """Две модели, изотоническая калибровка на отдельном периоде, логически согласованное объединение."""

    def __init__(self, features, quiet_target="y_quiet", any_target="y_any", iterations=300):
        self.features, self.quiet_target, self.any_target = features, quiet_target, any_target
        self.iterations = iterations

    def frame(self, df):
        x = df[self.features].copy()
        for col in cat_indices(self.features):
            x[col] = x[col].astype(str)
        return x

    def fit(self, train, cal):
        self.models, self.calibrators = {}, {}
        x, xc = self.frame(train), self.frame(cal)
        for name, target in [("close", "y_close"), ("quiet", self.quiet_target)]:
            model = CatBoostClassifier(iterations=self.iterations, depth=5, learning_rate=.06, loss_function="Logloss",
                                       random_seed=SEED, thread_count=6, verbose=False, allow_writing_files=False)
            model.fit(x, train[target], cat_features=cat_indices(self.features))
            self.models[name] = model
            self.calibrators[name] = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(xc)[:, 1], cal[target])
        self.union = IsotonicRegression(out_of_bounds="clip").fit(self._raw(xc), cal[self.any_target])
        return self

    def _raw(self, x):
        return sum(m.predict_proba(x)[:, 1] for m in self.models.values())

    def predict(self, df):
        x = self.frame(df)
        out = {k: self.calibrators[k].predict(m.predict_proba(x)[:, 1]) for k, m in self.models.items()}
        out["any"] = np.maximum(self.union.predict(self._raw(x)), np.maximum(out["close"], out["quiet"]))
        return out


class LogisticBaseline:
    """Простой честный бейзлайн на тех же признаках v2."""

    def __init__(self, target="y_any"):
        self.target = target
        numeric = [c for c in FEATURES_V2 if c not in CATEGORICAL]
        columns = make_column_transformer((StandardScaler(), numeric),
                                          (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL))
        self.pipe = make_pipeline(columns, LogisticRegression(max_iter=300, C=1.))

    def fit(self, train, cal):
        self.pipe.fit(train[FEATURES_V2], train[self.target])
        self.cal = IsotonicRegression(out_of_bounds="clip").fit(
            self.pipe.predict_proba(cal[FEATURES_V2])[:, 1], cal[self.target])
        return self

    def predict(self, df):
        return self.cal.predict(self.pipe.predict_proba(df[FEATURES_V2])[:, 1])


class RuleBaseline:
    """Правила Gate 3: вероятности по калибровочному периоду, равные уровни — фиксированный случайный порядок."""

    def fit(self, cal, target="y_any"):
        score = rule_score(cal)
        self.probability = {float(k): float(cal.loc[score == k, target].mean()) for k in np.unique(score)}
        self.fallback = float(cal[target].mean())
        return self

    def predict(self, df):
        score = rule_score(df)
        return np.array([self.probability.get(k, self.fallback) for k in score])


class UpliftModel:
    """Многоветочный T-learner. Обучается только на наблюдаемом исходе своей случайной ветви."""

    def __init__(self, features=FEATURES_V2, target="retained365"):
        self.features, self.target = features, target

    def frame(self, df):
        x = df[self.features].copy()
        for col in cat_indices(self.features):
            x[col] = x[col].astype(str)
        return x

    def fit(self, train):
        self.models = {}
        for arm in range(4):
            part = train[train.arm == arm]
            model = CatBoostClassifier(iterations=320, depth=4, learning_rate=.05, l2_leaf_reg=8, loss_function="Logloss",
                                       random_seed=SEED + arm, thread_count=6, verbose=False, allow_writing_files=False)
            model.fit(self.frame(part), part[self.target], cat_features=cat_indices(self.features))
            self.models[arm] = model
        return self

    def predict(self, df):
        x = self.frame(df)
        control = self.models[0].predict_proba(x)[:, 1]
        return np.stack([self.models[a].predict_proba(x)[:, 1] - control for a in [1, 2, 3]], axis=1)


def assert_no_forbidden(features):
    bad = FORBIDDEN & set(features)
    if bad:
        raise ValueError(f"Forbidden features: {sorted(bad)}")
