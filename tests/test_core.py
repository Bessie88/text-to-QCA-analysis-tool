from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qca_text_tool.calibration import calibrate_scores
from qca_text_tool.qca import solution_configurations, truth_table
from qca_text_tool.scoring import (
    add_low_signal_floor,
    score_texts_against_prototypes,
    wide_score_table,
)


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
