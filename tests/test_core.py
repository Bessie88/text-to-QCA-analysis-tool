from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qca_text_tool.calibration import calibrate_scores
from qca_text_tool.qca import solution_configurations, truth_table
from qca_text_tool.scoring import (
    _keyword_match_score,
    add_low_signal_floor,
    score_texts_against_prototypes,
    wide_score_table,
)


class KeywordMatchingTest(unittest.TestCase):
    def test_keyword_matching_avoids_substring_and_negation_false_positives(self):
        score, matched = _keyword_match_score(
            "Residents distrust the current explanation.",
            ("trust", "distrust"),
        )
        self.assertEqual(score, 0.5)
        self.assertEqual(matched, "distrust")

        score, matched = _keyword_match_score(
            "新规定说法前后不一致，居民不信任目前的解释。",
            ("信任", "不信任"),
        )
        self.assertEqual(score, 0.5)
        self.assertEqual(matched, "不信任")

        score, matched = _keyword_match_score("居民信任并支持这项工作。", ("信任",))
        self.assertEqual(score, 1.0)
        self.assertEqual(matched, "信任")


class DemoWorkflowTest(unittest.TestCase):
    def test_end_to_end_demo_workflow(self):
        texts = pd.read_csv(PROJECT_ROOT / "data" / "demo_texts.csv")
        prototypes = pd.read_csv(PROJECT_ROOT / "data" / "prototypes.csv")
        scores = score_texts_against_prototypes(
            texts,
            prototypes,
            case_col="case_id",
            text_col="text",
        )
        self.assertFalse(scores.empty)
        self.assertTrue(scores["score"].between(0, 1).all())

        scores = add_low_signal_floor(scores)
        wide = wide_score_table(scores, value_col="calibration_score")
        condition_cols = prototypes.loc[
            prototypes["type"].str.lower().eq("condition"), "condition_name"
        ].tolist()
        calibrated, rules = calibrate_scores(wide, condition_cols, method="fuzzy")
        self.assertTrue(set(condition_cols).issubset(calibrated.columns))
        self.assertFalse(rules.empty)

        qca_ready = calibrated.merge(texts[["case_id", "outcome"]], on="case_id")
        table = truth_table(qca_ready, condition_cols, "outcome")
        solutions = solution_configurations(table)
        self.assertFalse(table.empty)
        self.assertTrue({"consistency", "coverage", "configuration"}.issubset(table.columns))
        self.assertFalse(solutions.empty)


if __name__ == "__main__":
    unittest.main()
