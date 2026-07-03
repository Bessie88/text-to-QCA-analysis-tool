"""Small, transparent QCA calculations used by the Streamlit app."""

from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
import pandas as pd


def _numeric_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result


def format_configuration(config: dict[str, int], condition_cols: list[str]) -> str:
    parts = []
    for condition in condition_cols:
        value = int(config[condition])
        parts.append(condition if value == 1 else f"~{condition}")
    return " * ".join(parts)


def configuration_membership(
    df: pd.DataFrame, config: dict[str, int], condition_cols: list[str]
) -> pd.Series:
    memberships = []
    for condition in condition_cols:
        values = pd.to_numeric(df[condition], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        memberships.append(values if int(config[condition]) == 1 else 1.0 - values)
    if not memberships:
        return pd.Series(0.0, index=df.index)
    return pd.concat(memberships, axis=1).min(axis=1)


def consistency_coverage(
    config_membership: pd.Series, outcome: pd.Series
) -> tuple[float, float]:
    x = pd.to_numeric(config_membership, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    y = pd.to_numeric(outcome, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    overlap = np.minimum(x, y)
    consistency_denominator = float(x.sum())
    coverage_denominator = float(y.sum())
    consistency = 0.0 if consistency_denominator == 0 else float(overlap.sum() / consistency_denominator)
    coverage = 0.0 if coverage_denominator == 0 else float(overlap.sum() / coverage_denominator)
    return consistency, coverage


def truth_table(
    qca_df: pd.DataFrame,
    condition_cols: list[str],
    outcome_col: str,
    case_col: str = "case_id",
    binarize_threshold: float = 0.5,
    consistency_cutoff: float = 0.8,
    min_cases: int = 1,
    include_unobserved: bool = False,
) -> pd.DataFrame:
    """Build a truth table with standard consistency and coverage metrics."""
    required = set(condition_cols + [outcome_col])
    if case_col in qca_df.columns:
        required.add(case_col)
    missing = required.difference(qca_df.columns)
    if missing:
        raise ValueError(f"QCA dataset is missing required columns: {sorted(missing)}")

    data = _numeric_frame(qca_df, condition_cols + [outcome_col])
    binary_conditions = (data[condition_cols] >= binarize_threshold).astype(int)
    observed_configs = binary_conditions.drop_duplicates().to_dict("records")
    if include_unobserved:
        observed_configs = [
            dict(zip(condition_cols, values))
            for values in itertools.product([0, 1], repeat=len(condition_cols))
        ]

    rows = []
    for config in observed_configs:
        mask = pd.Series(True, index=data.index)
        for condition, value in config.items():
            mask &= binary_conditions[condition] == int(value)
        subset = data.loc[mask]
        case_ids = (
            qca_df.loc[mask, case_col].astype(str).tolist()
            if case_col in qca_df.columns
            else subset.index.astype(str).tolist()
        )
        config_x = configuration_membership(data, config, condition_cols)
        consistency, coverage = consistency_coverage(config_x, data[outcome_col])
        positive_cases = int((subset[outcome_col] >= binarize_threshold).sum())
        negative_cases = int((subset[outcome_col] < binarize_threshold).sum())
        n_cases = int(mask.sum())
        outcome_value = (
            1
            if n_cases >= int(min_cases) and consistency >= float(consistency_cutoff)
            else 0
        )
        if n_cases == 0:
            status = "unobserved"
        elif positive_cases > 0 and negative_cases > 0:
            status = "contradictory"
        elif consistency >= float(consistency_cutoff):
            status = "sufficient"
        elif consistency >= 0.5:
            status = "weak"
        else:
            status = "not_sufficient"

        row = {condition: int(config[condition]) for condition in condition_cols}
        row.update(
            {
                "configuration": format_configuration(config, condition_cols),
                "n_cases": n_cases,
                "positive_cases": positive_cases,
                "negative_cases": negative_cases,
                "consistency": round(consistency, 6),
                "coverage": round(coverage, 6),
                "outcome_value": outcome_value,
                "status": status,
                "cases": "; ".join(case_ids),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["outcome_value", "consistency", "coverage", "n_cases"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def solution_configurations(
    truth_df: pd.DataFrame,
    consistency_cutoff: float = 0.8,
    min_cases: int = 1,
) -> pd.DataFrame:
    """Return observed configurations that meet the sufficiency threshold."""
    if truth_df.empty:
        return truth_df.copy()
    required = {"configuration", "n_cases", "consistency", "coverage"}
    missing = required.difference(truth_df.columns)
    if missing:
        raise ValueError(f"Truth table is missing required columns: {sorted(missing)}")
    solution = truth_df[
        (truth_df["n_cases"] >= int(min_cases))
        & (truth_df["consistency"] >= float(consistency_cutoff))
        & (truth_df.get("status", "sufficient") == "sufficient")
    ].copy()
    return solution.sort_values(
        ["consistency", "coverage", "n_cases"], ascending=[False, False, False]
    ).reset_index(drop=True)


def consistency_cutoff_sweep(
    qca_df: pd.DataFrame,
    condition_cols: list[str],
    outcome_col: str,
    case_col: str = "case_id",
    binarize_threshold: float = 0.5,
    cutoffs: Iterable[float] | None = None,
    min_cases: int = 1,
) -> pd.DataFrame:
    """Report how the solution set changes as the consistency cutoff varies.

    Raising the cutoff can only keep or shrink the solution set, since sufficiency
    becomes a stricter test; this makes the choice of cutoff, and its consequences,
    visible instead of baked silently into a single reported result.
    """
    if cutoffs is None:
        cutoffs = [round(value, 2) for value in np.arange(0.5, 0.96, 0.05)]
    rows = []
    for cutoff in cutoffs:
        table = truth_table(
            qca_df,
            condition_cols,
            outcome_col,
            case_col,
            binarize_threshold=binarize_threshold,
            consistency_cutoff=cutoff,
            min_cases=min_cases,
        )
        solutions = solution_configurations(table, consistency_cutoff=cutoff, min_cases=min_cases)
        rows.append(
            {
                "consistency_cutoff": round(float(cutoff), 3),
                "n_solutions": len(solutions),
                "avg_solution_consistency": float(solutions["consistency"].mean())
                if not solutions.empty
                else np.nan,
                "avg_solution_coverage": float(solutions["coverage"].mean())
                if not solutions.empty
                else np.nan,
                "n_contradictory": int((table["status"] == "contradictory").sum())
                if not table.empty
                else 0,
            }
        )
    return pd.DataFrame(rows)


def binarize_threshold_sweep(
    qca_df: pd.DataFrame,
    condition_cols: list[str],
    outcome_col: str,
    case_col: str = "case_id",
    consistency_cutoff: float = 0.8,
    thresholds: Iterable[float] | None = None,
    min_cases: int = 1,
) -> pd.DataFrame:
    """Report how truth-table configurations change as the crossover used to
    binarize membership into configurations varies. Different crossovers can
    reassign cases to different configurations, which is a second, independent
    source of sensitivity beyond the consistency cutoff.
    """
    if thresholds is None:
        thresholds = [round(value, 2) for value in np.arange(0.3, 0.71, 0.05)]
    rows = []
    for threshold in thresholds:
        table = truth_table(
            qca_df,
            condition_cols,
            outcome_col,
            case_col,
            binarize_threshold=threshold,
            consistency_cutoff=consistency_cutoff,
            min_cases=min_cases,
        )
        solutions = solution_configurations(
            table, consistency_cutoff=consistency_cutoff, min_cases=min_cases
        )
        rows.append(
            {
                "binarize_threshold": round(float(threshold), 3),
                "n_configurations": len(table),
                "n_solutions": len(solutions),
                "avg_solution_consistency": float(solutions["consistency"].mean())
                if not solutions.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
