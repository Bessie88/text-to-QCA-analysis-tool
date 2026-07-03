"""Core helpers for the text-to-QCA take-home tool."""

from .calibration import calibrate_scores
from .qca import solution_configurations, truth_table
from .scoring import add_low_signal_floor, score_texts_against_prototypes, wide_score_table

__all__ = [
    "add_low_signal_floor",
    "calibrate_scores",
    "score_texts_against_prototypes",
    "solution_configurations",
    "truth_table",
    "wide_score_table",
]
