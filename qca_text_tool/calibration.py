"""Calibration from raw text scores to crisp or fuzzy set memberships."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def default_fuzzy_anchors(values: pd.Series) -> tuple[float, float, float]:
    clean = _safe_numeric(values).dropna()
    if clean.empty:
        return 0.0, 0.5, 1.0

    full_out = float(clean.quantile(0.25))
    crossover = float(clean.quantile(0.50))
    full_in = float(clean.quantile(0.75))
    if full_in > full_out:
        if crossover <= full_out or crossover >= full_in:
            crossover = full_out + ((full_in - full_out) / 2.0)
        return full_out, crossover, full_in

    min_value = float(clean.min())
    max_value = float(clean.max())
    if max_value > min_value:
        return min_value, float(clean.median()), max_value

    center = float(clean.iloc[0])
    return max(0.0, center - 0.001), center, min(1.0, center + 0.001)


def fuzzy_membership(
    values: pd.Series, full_out: float, crossover: float, full_in: float
) -> pd.Series:
    numeric = _safe_numeric(values)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if full_in <= full_out:
        result.loc[numeric.notna()] = 0.5
        return result

    crossover = min(max(crossover, full_out), full_in)
    below = numeric <= full_out
    above = numeric >= full_in
    middle_low = (numeric > full_out) & (numeric < crossover)
    middle_high = (numeric >= crossover) & (numeric < full_in)

    result.loc[below] = 0.0
    result.loc[above] = 1.0
    if crossover > full_out:
        result.loc[middle_low] = (
            0.5 * (numeric.loc[middle_low] - full_out) / (crossover - full_out)
        )
    else:
        result.loc[middle_low] = 0.5
    if full_in > crossover:
        result.loc[middle_high] = 0.5 + (
            0.5 * (numeric.loc[middle_high] - crossover) / (full_in - crossover)
        )
    else:
        result.loc[middle_high] = 0.5
    return result.clip(0.0, 1.0)


def crisp_membership(values: pd.Series, threshold: float) -> pd.Series:
    numeric = _safe_numeric(values)
    return (numeric >= threshold).astype(int)


def calibrate_scores(
    score_wide: pd.DataFrame,
    condition_cols: list[str],
    method: str = "fuzzy",
    anchors: dict[str, tuple[float, float, float]] | None = None,
    crisp_thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate each condition score column.

    Returns a calibrated membership table and a table describing the anchors or
    thresholds used for reproducibility.
    """
    method = method.lower().strip()
    if method not in {"fuzzy", "crisp"}:
        raise ValueError("Calibration method must be either 'fuzzy' or 'crisp'.")
    missing = [col for col in condition_cols if col not in score_wide.columns]
    if missing:
        raise ValueError(f"Score table is missing condition columns: {missing}")

    anchors = anchors or {}
    crisp_thresholds = crisp_thresholds or {}
    calibrated = pd.DataFrame(index=score_wide.index)
    if "case_id" in score_wide.columns:
        calibrated["case_id"] = score_wide["case_id"]

    metadata_rows = []
    for condition in condition_cols:
        values = _safe_numeric(score_wide[condition])
        if method == "fuzzy":
            full_out, crossover, full_in = anchors.get(
                condition, default_fuzzy_anchors(values)
            )
            calibrated[condition] = fuzzy_membership(
                values, full_out=full_out, crossover=crossover, full_in=full_in
            ).round(6)
            metadata_rows.append(
                {
                    "condition_name": condition,
                    "method": "fuzzy",
                    "full_out": round(float(full_out), 6),
                    "crossover": round(float(crossover), 6),
                    "full_in": round(float(full_in), 6),
                    "threshold": None,
                }
            )
        else:
            threshold = crisp_thresholds.get(condition, float(values.median()))
            calibrated[condition] = crisp_membership(values, threshold)
            metadata_rows.append(
                {
                    "condition_name": condition,
                    "method": "crisp",
                    "full_out": None,
                    "crossover": None,
                    "full_in": None,
                    "threshold": round(float(threshold), 6),
                }
            )

    return calibrated, pd.DataFrame(metadata_rows)


def near_threshold_cases(
    score_wide: pd.DataFrame,
    calibration_rules: pd.DataFrame,
    condition_cols: list[str],
    band_fraction: float = 0.1,
) -> pd.DataFrame:
    """Flag (case, condition) pairs whose raw score sits close to the calibration anchor.

    "Close" means within ``band_fraction`` of the condition's raw-score range from the
    fuzzy crossover (or the crisp threshold). These are the cases where a small change
    in wording, or a small change in the anchor, could flip the calibrated membership --
    good candidates for a researcher to review by hand rather than trust automatically.
    """
    if calibration_rules.empty:
        return pd.DataFrame(
            columns=["case_id", "condition_name", "raw_score", "anchor", "distance_fraction"]
        )
    rules = calibration_rules.set_index("condition_name")
    rows = []
    for column in condition_cols:
        if column not in score_wide.columns or column not in rules.index:
            continue
        values = _safe_numeric(score_wide[column])
        score_range = float(values.max() - values.min())
        if score_range <= 0:
            continue
        rule = rules.loc[column]
        anchor = rule.get("crossover")
        if pd.isna(anchor):
            anchor = rule.get("threshold")
        if pd.isna(anchor):
            continue
        anchor = float(anchor)
        distance_fraction = (values - anchor).abs() / score_range
        flagged = distance_fraction <= band_fraction
        case_ids = score_wide["case_id"]
        for case_id, is_flagged, raw_value, distance in zip(
            case_ids, flagged, values, distance_fraction
        ):
            if not bool(is_flagged) or pd.isna(raw_value):
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "condition_name": column,
                    "raw_score": round(float(raw_value), 6),
                    "anchor": round(anchor, 6),
                    "distance_fraction": round(float(distance), 6),
                }
            )
    return pd.DataFrame(
        rows, columns=["case_id", "condition_name", "raw_score", "anchor", "distance_fraction"]
    )
