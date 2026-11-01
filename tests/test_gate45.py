"""Структурные проверки: не заменяют eval-сет, а защищают код от регрессий."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import CAL_CUTS, DEV, TEST_CUTS, TRAIN_CUTS  # noqa: E402
from dataclasses import replace  # noqa: E402
from generator import add_features, generate, split_cuts_are_valid  # noqa: E402
from labeling import agreement, expand, heuristic  # noqa: E402
from legacy import OFFERS, eligibility, quote, quote_vec  # noqa: E402
from metrics import power_total, summary, top_mask, wilson  # noqa: E402
from models import FEATURES_G3, FEATURES_V2, FORBIDDEN  # noqa: E402
from service import MIN_BATCH, batch_gate, row_codes  # noqa: E402


class Generator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, df, cls.truth = generate(replace(DEV, n=6000))
        cls.df, cls.profile = add_features(cls.panel.copy(), df)

    def test_split_dates(self):
        self.assertTrue(split_cuts_are_valid())
        self.assertLess(max(TRAIN_CUTS), min(CAL_CUTS))
        self.assertLess(max(CAL_CUTS), min(TEST_CUTS))

    def test_clients_do_not_repeat(self):
        self.assertTrue(self.df.client_id.is_unique)

    def test_features_do_not_read_future(self):
        changed = self.panel.copy()
        idx = np.arange(len(self.df))
        cut = self.df.cut.to_numpy()
        for j in range(len(idx)):
            changed[j, cut[j]:] = 1
        again, _ = add_features(changed, self.df.drop(columns=["turnover_ratio"]).copy(), self.profile)
        for col in ["turnover_ratio", "season_adjusted_ratio", "yoy_ratio", "yoy_ratio_3m", "last_year_forward_ratio"]:
            np.testing.assert_array_equal(again[col].to_numpy(), self.df[col].to_numpy())

    def test_control_outcome_matches_churn_time(self):
        np.testing.assert_array_equal(self.truth.keep365_0.to_numpy(), self.truth.t0.to_numpy() > 365)

    def test_adjusted_target_is_subset(self):
        self.assertTrue((self.df.y_any_adj <= self.df.y_any).all())


class Calculator(unittest.TestCase):
    def test_vector_matches_scalar(self):
        payments = np.arange(0, 240, 7)
        for tariff in ["advanced", "professional"]:
            vector = quote_vec(np.full(len(payments), tariff), payments)
            for i, p in enumerate(payments):
                for o in OFFERS:
                    q = quote(tariff, p, o)
                    self.assertEqual(int(vector[o]["cost"][i]), q["budget_cost"])
                    self.assertEqual(bool(vector[o]["allowed"][i]), q["allowed"])

    def test_no_allowed_offer_raises_price(self):
        for tariff in ["advanced", "professional"]:
            for p in range(0, 300, 5):
                for o in OFFERS:
                    q = quote(tariff, p, o)
                    if q["allowed"]:
                        self.assertLessEqual(q["after_year"], q["before_year"])
                        self.assertLessEqual(q["after_month"], q["current_month"])


class Metrics(unittest.TestCase):
    def test_top_is_inside_month(self):
        month = np.array([1] * 100 + [2] * 100)
        score = np.r_[np.arange(100), np.arange(100) + 1000.]
        mask = top_mask(score, month, .1)
        self.assertEqual(mask[:100].sum(), 10)
        self.assertEqual(mask[100:].sum(), 10)

    def test_lift_definition(self):
        y = np.r_[np.ones(10), np.zeros(90)]
        out = summary(y, -np.arange(100.), np.zeros(100))
        self.assertAlmostEqual(out["precision"], 1)
        self.assertAlmostEqual(out["lift"], 10)

    def test_power_matches_gate3(self):
        self.assertAlmostEqual(power_total(.75, .10), 1583, delta=1)

    def test_wilson_contains_point(self):
        low, high = wilson(40, 100)
        self.assertTrue(low < .4 < high)


class Service(unittest.TestCase):
    def test_forbidden_fields_absent(self):
        self.assertFalse(FORBIDDEN & set(FEATURES_V2))
        self.assertIn("turnover_ratio", FEATURES_G3)
        self.assertNotIn("turnover_ratio", FEATURES_V2)
        self.assertNotIn("region", FEATURES_V2)

    @classmethod
    def setUpClass(cls):
        panel, df, _ = generate(replace(DEV, n=6000))
        cls.df, _ = add_features(panel, df)

    def test_small_batch_is_not_blocked_by_psi(self):
        selected = self.df.sort_values("fee_burden").tail(MIN_BATCH - 1)
        self.assertFalse(batch_gate(selected, self.df)["blocked"])

    def test_large_shifted_batch_is_blocked(self):
        shifted = self.df.head(MIN_BATCH).assign(fee_burden=lambda d: d.fee_burden + 1.5)
        self.assertTrue(batch_gate(shifted, self.df)["blocked"])

    def test_compliance_has_priority_over_escalation(self):
        frame = self.df.head(20).assign(compliance=True, revenue_year=5_000_000)
        self.assertTrue(row_codes(frame).isin(["EXCLUDE_COMPLIANCE", "REJECT_SCHEMA"]).all())

    def test_unknown_category_is_rejected(self):
        self.assertTrue(row_codes(self.df.head(20).assign(industry="agro")).eq("REJECT_SCHEMA").all())


class Labeling(unittest.TestCase):
    def test_labels_cover_every_case(self):
        cases = pd.read_csv(ROOT / "labeling" / "cases_generated.csv")
        labels = expand(ROOT / "labeling" / "labels_round1_compact.txt")
        self.assertEqual(set(cases.id), set(labels.id))
        self.assertTrue(cases.text.is_unique)

    def test_action_is_consistent(self):
        labels = expand(ROOT / "labeling" / "labels_round1_compact.txt")
        self.assertTrue(labels.loc[labels.manageable != "yes", "action"].eq("none").all())
        self.assertTrue(labels.loc[labels.manageable == "yes", "action"].ne("none").all())
        self.assertTrue(labels.loc[labels.reason == "forced", "manageable"].eq("no").all())

    def test_heuristic_excludes_forced_markers(self):
        self.assertEqual(heuristic(["Счёт заблокировали, комиссия ни при чём"]).tolist(), [0])


if __name__ == "__main__":
    unittest.main()
