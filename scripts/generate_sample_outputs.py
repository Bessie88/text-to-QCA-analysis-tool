from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qca_text_tool.calibration import calibrate_scores, near_threshold_cases
from qca_text_tool.qca import (
    binarize_threshold_sweep,
    consistency_cutoff_sweep,
    solution_configurations,
    truth_table,
)
from qca_text_tool.scoring import score_texts_against_prototypes, wide_score_table
from qca_text_tool.scoring import add_low_signal_floor


DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def write_membership_heatmap(qca_ready: pd.DataFrame, output_path: Path) -> None:
    condition_cols = [col for col in qca_ready.columns if col not in {"case_id"}]
    cell_size = 28
    label_width = 86
    top_margin = 130
    width = label_width + cell_size * len(condition_cols) + 40
    height = top_margin + cell_size * len(qca_ready) + 45

    def color(value: float) -> str:
        value = max(0.0, min(1.0, float(value)))
        red = int(245 - 170 * value)
        green = int(247 - 75 * value)
        blue = int(250 - 180 * value)
        return f"rgb({red},{green},{blue})"

    cells = []
    for row_idx, row in qca_ready.iterrows():
        y = top_margin + row_idx * cell_size
        cells.append(
            f'<text x="8" y="{y + 18}" font-size="11" fill="#263238">{row["case_id"]}</text>'
        )
        for col_idx, col in enumerate(condition_cols):
            x = label_width + col_idx * cell_size
            value = float(row[col])
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="{color(value)}" stroke="#ffffff" stroke-width="1"/>'
            )
            cells.append(
                f'<text x="{x + cell_size / 2}" y="{y + 18}" font-size="9" '
                f'text-anchor="middle" fill="#111827">{value:.2f}</text>'
            )

    headers = []
    for col_idx, col in enumerate(condition_cols):
        x = label_width + col_idx * cell_size + 18
        headers.append(
            f'<text transform="translate({x},{top_margin - 8}) rotate(-55)" '
            f'font-size="11" fill="#263238">{col}</text>'
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Membership Heatmap</title>
</head>
<body style="font-family: Arial, sans-serif; margin: 24px;">
  <h2 style="margin: 0 0 4px;">Membership Heatmap</h2>
  <p style="margin: 0 0 16px; color: #4b5563;">Darker cells indicate stronger set membership.</p>
  <svg width="{width}" height="{height}" role="img" aria-label="Membership heatmap">
    <text x="8" y="22" font-size="14" font-weight="700" fill="#111827">Case</text>
    {''.join(headers)}
    {''.join(cells)}
  </svg>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    texts = pd.read_csv(DATA_DIR / "demo_texts.csv")
    prototypes = pd.read_csv(DATA_DIR / "prototypes.csv")

    scores = score_texts_against_prototypes(
        texts,
        prototypes,
        case_col="case_id",
        text_col="text",
        keyword_weight=0.35,
    )
    scores = add_low_signal_floor(scores)
    score_wide = wide_score_table(scores, value_col="calibration_score")

    condition_cols = prototypes.loc[
        prototypes["type"].str.lower().eq("condition"), "condition_name"
    ].tolist()
    # Substantive override: the default 75th-percentile crossover (~0.176) sits
    # below cooperative/grateful cases whose semantic score is inflated by
    # shared "citizen-government" phrasing despite not being complaints. 0.22
    # sits at the natural gap between the mildest genuine complaint (case 2,
    # 0.230) and the strongest false-positive cooperative case (case 8, 0.214).
    anchors = {"dissatisfaction": (0.134692, 0.22, 0.30)}
    calibrated, calibration_rules = calibrate_scores(
        score_wide,
        condition_cols=condition_cols,
        method="fuzzy",
        anchors=anchors,
    )

    qca_ready = calibrated.merge(texts[["case_id", "outcome"]], on="case_id", how="left")
    table = truth_table(
        qca_ready,
        condition_cols=condition_cols,
        outcome_col="outcome",
        case_col="case_id",
        consistency_cutoff=0.8,
        min_cases=1,
    )
    solutions = solution_configurations(table, consistency_cutoff=0.8, min_cases=1)

    near_threshold = near_threshold_cases(
        score_wide, calibration_rules, condition_cols, band_fraction=0.1
    )
    if not near_threshold.empty:
        near_threshold = near_threshold.merge(
            texts[["case_id", "text"]], on="case_id", how="left"
        )

    cutoff_sweep = consistency_cutoff_sweep(qca_ready, condition_cols, "outcome")
    threshold_sweep = binarize_threshold_sweep(qca_ready, condition_cols, "outcome")

    scores.to_csv(OUTPUT_DIR / "score_table.csv", index=False)
    score_wide.to_csv(OUTPUT_DIR / "classification_score_wide.csv", index=False)
    calibrated.to_csv(OUTPUT_DIR / "calibrated_membership.csv", index=False)
    calibration_rules.to_csv(OUTPUT_DIR / "calibration_rules.csv", index=False)
    qca_ready.to_csv(OUTPUT_DIR / "qca_ready_dataset.csv", index=False)
    table.to_csv(OUTPUT_DIR / "truth_table.csv", index=False)
    solutions.to_csv(OUTPUT_DIR / "solution_configurations.csv", index=False)
    near_threshold.to_csv(OUTPUT_DIR / "human_review_candidates.csv", index=False)
    cutoff_sweep.to_csv(OUTPUT_DIR / "sensitivity_cutoff_sweep.csv", index=False)
    threshold_sweep.to_csv(OUTPUT_DIR / "sensitivity_threshold_sweep.csv", index=False)
    write_membership_heatmap(qca_ready, OUTPUT_DIR / "membership_heatmap.html")


if __name__ == "__main__":
    main()
