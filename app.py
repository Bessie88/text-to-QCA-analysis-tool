from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from qca_text_tool.calibration import calibrate_scores, default_fuzzy_anchors, near_threshold_cases
from qca_text_tool.qca import (
    binarize_threshold_sweep,
    consistency_cutoff_sweep,
    solution_configurations,
    truth_table,
)
from qca_text_tool.scoring import (
    add_low_signal_floor,
    explain_score,
    highlight_matches,
    score_texts_against_prototypes,
    wide_score_table,
)

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

COLOR_BLUE = "#2a78d6"
COLOR_BLUE_DARK = "#184f95"
COLOR_ORANGE = "#eb6834"
COLOR_MUTED = "#9aa0a6"
COLOR_GOOD = "#0ca30c"
COLOR_WARNING = "#c98500"
COLOR_CRITICAL = "#d03b3b"

STATUS_COLORS = {
    "sufficient": COLOR_GOOD,
    "weak": COLOR_WARNING,
    "contradictory": COLOR_CRITICAL,
    "not_sufficient": COLOR_MUTED,
    "unobserved": "#c3c2b7",
}

BLUE_SEQUENTIAL_SCALE = [
    [0.0, "#eef2f8"],
    [0.25, "#9ec5f4"],
    [0.5, "#2a78d6"],
    [0.75, "#1c5cab"],
    [1.0, "#0d366b"],
]

CUSTOM_CSS = f"""
<style>
.qca-hero {{
    background: linear-gradient(135deg, {COLOR_BLUE} 0%, {COLOR_BLUE_DARK} 100%);
    padding: 28px 32px;
    border-radius: 14px;
    color: #ffffff;
    margin-bottom: 18px;
}}
.qca-hero h1 {{
    margin: 0 0 6px 0;
    font-size: 1.85rem;
    font-weight: 700;
}}
.qca-hero p {{
    margin: 0;
    opacity: 0.92;
    font-size: 0.98rem;
}}
div[data-testid="stMetric"] {{
    background: #ffffff;
    border: 1px solid #e1e0d9;
    border-radius: 10px;
    padding: 12px 14px 8px 14px;
    box-shadow: 0 1px 3px rgba(11, 11, 11, 0.06);
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    font-weight: 600;
    padding: 8px 14px;
}}
mark {{
    background: #ffd68a;
    color: #2b1a00;
    padding: 0 3px;
    border-radius: 3px;
}}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #f7f9fc 0%, #eef2f8 100%);
    border-right: 1px solid #e1e0d9;
}}
section[data-testid="stSidebar"] .qca-sidebar-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {COLOR_BLUE_DARK};
    margin: 2px 0 2px 0;
}}
section[data-testid="stSidebar"] .qca-sidebar-caption {{
    font-size: 0.82rem;
    color: #52514e;
    margin-bottom: 6px;
}}
section[data-testid="stSidebar"] h2 {{
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {COLOR_BLUE_DARK};
    border-bottom: 2px solid {COLOR_BLUE};
    padding-bottom: 6px;
    margin-top: 22px;
    margin-bottom: 10px;
}}
section[data-testid="stSidebar"] .stButton button {{
    background: #ffffff;
    border: 1px solid {COLOR_CRITICAL};
    color: {COLOR_CRITICAL};
    border-radius: 8px;
    font-weight: 600;
    width: 100%;
}}
section[data-testid="stSidebar"] .stButton button:hover {{
    background: {COLOR_CRITICAL};
    color: #ffffff;
    border-color: {COLOR_CRITICAL};
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    border-radius: 10px;
}}
</style>
"""


def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(DATA_DIR / "demo_texts.csv"),
        pd.read_csv(DATA_DIR / "prototypes.csv"),
    )


def download_button(label: str, df: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )


def coerce_outcome(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map(
        {
            "1": 1.0,
            "0": 0.0,
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "positive": 1.0,
            "negative": 0.0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    result = numeric.where(numeric.notna(), mapped)
    if result.max(skipna=True) and result.max(skipna=True) > 1:
        result = result / result.max(skipna=True)
    return result.fillna(0.0).clip(0.0, 1.0)


def collect_calibration_controls(
    score_wide: pd.DataFrame, calibration_cols: list[str], method: str
) -> tuple[dict[str, tuple[float, float, float]], dict[str, float]]:
    anchors: dict[str, tuple[float, float, float]] = {}
    thresholds: dict[str, float] = {}
    with st.sidebar.expander("🎯 Edit calibration anchors", expanded=False):
        for column in calibration_cols:
            values = pd.to_numeric(score_wide[column], errors="coerce").dropna()
            if values.empty:
                continue
            min_score = float(values.min())
            max_score = float(values.max())
            if max_score <= min_score:
                st.caption(f"{column}: all raw scores are identical.")
                continue

            if method == "fuzzy":
                full_out, crossover, full_in = default_fuzzy_anchors(values)
                step = max((max_score - min_score) / 100.0, 0.000001)
                selected_full_out = st.slider(
                    f"{column}: full out",
                    min_value=min_score,
                    max_value=max_score,
                    value=max(min_score, min(float(full_out), max_score)),
                    step=step,
                    key=f"full_out_{column}",
                )
                selected_crossover = st.slider(
                    f"{column}: crossover",
                    min_value=min_score,
                    max_value=max_score,
                    value=max(min_score, min(float(crossover), max_score)),
                    step=step,
                    key=f"crossover_{column}",
                )
                selected_full_in = st.slider(
                    f"{column}: full in",
                    min_value=min_score,
                    max_value=max_score,
                    value=max(min_score, min(float(full_in), max_score)),
                    step=step,
                    key=f"full_in_{column}",
                )
                ordered = tuple(
                    sorted(
                        [
                            float(selected_full_out),
                            float(selected_crossover),
                            float(selected_full_in),
                        ]
                    )
                )
                anchors[column] = ordered
            else:
                threshold = float(values.median())
                thresholds[column] = st.slider(
                    column,
                    min_value=min_score,
                    max_value=max_score,
                    value=max(min_score, min(threshold, max_score)),
                    step=max((max_score - min_score) / 100.0, 0.000001),
                    key=f"threshold_{column}",
                )
    return anchors, thresholds


def apply_human_overrides(
    calibrated: pd.DataFrame, overrides: dict[tuple[object, str], float]
) -> pd.DataFrame:
    reviewed = calibrated.copy()
    for (case_id, condition_name), value in overrides.items():
        if condition_name not in reviewed.columns:
            continue
        mask = reviewed["case_id"] == case_id
        reviewed.loc[mask, condition_name] = value
    return reviewed


def main() -> None:
    st.set_page_config(page_title="Text-to-QCA Analysis Tool", layout="wide", page_icon="🧭")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="qca-hero">
            <h1>🧭 Text-to-QCA Analysis Tool</h1>
            <p>Turn raw text into calibrated conditions and a QCA solution — upload, score, calibrate, solve.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    demo_texts, demo_prototypes = load_demo_data()

    with st.sidebar:
        st.markdown('<div class="qca-sidebar-title">⚙️ Controls</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="qca-sidebar-caption">Adjust these to steer the pipeline — every tab updates live.</div>',
            unsafe_allow_html=True,
        )

        st.header("📂 Data")
        text_file = st.file_uploader("Text dataset CSV", type=["csv"])
        prototype_file = st.file_uploader("Prototype CSV", type=["csv"])
        texts = pd.read_csv(text_file) if text_file else demo_texts.copy()
        prototypes = (
            pd.read_csv(prototype_file) if prototype_file else demo_prototypes.copy()
        )

        case_col = st.selectbox(
            "Case id column",
            texts.columns.tolist(),
            index=texts.columns.tolist().index("case_id")
            if "case_id" in texts.columns
            else 0,
        )
        text_col = st.selectbox(
            "Text column",
            texts.columns.tolist(),
            index=texts.columns.tolist().index("text") if "text" in texts.columns else 0,
        )
        outcome_candidates = ["Use prototype outcome"] + [
            col for col in texts.columns if col not in {case_col, text_col}
        ]
        outcome_choice = st.selectbox(
            "Outcome source",
            outcome_candidates,
            index=outcome_candidates.index("outcome") if "outcome" in outcome_candidates else 0,
        )

        st.header("🧮 Scoring")
        min_ngram = st.slider("Minimum character n-gram", 1, 4, 2)
        max_ngram = st.slider("Maximum character n-gram", min_ngram, 6, 4)
        keyword_weight = st.slider("Keyword weight", 0.0, 1.0, 0.35, 0.05)

        st.header("🎚️ Calibration")
        calibration_method = st.radio("Set type", ["fuzzy", "crisp"], horizontal=True)

        st.header("🧭 QCA")
        consistency_cutoff = st.slider("Consistency cutoff", 0.5, 1.0, 0.8, 0.01)
        min_cases = st.number_input("Minimum cases per configuration", 1, 20, 1)
        binarize_threshold = st.slider("Configuration crossover", 0.1, 0.9, 0.5, 0.05)

        st.header("🔍 Human review")
        review_band = st.slider(
            "Ambiguity band (fraction of score range around the calibration anchor)",
            0.0,
            0.3,
            0.1,
            0.01,
            help="Cases whose raw score falls within this fraction of the anchor "
            "(crossover for fuzzy sets, threshold for crisp sets) are flagged as "
            "ambiguous and can be overridden by hand in the Human review tab.",
        )
        if st.button("Clear human review overrides"):
            st.session_state["human_overrides"] = {}

    if "human_overrides" not in st.session_state:
        st.session_state["human_overrides"] = {}

    required_prototype_cols = {"condition_name", "prototype"}
    missing_prototype_cols = required_prototype_cols.difference(prototypes.columns)
    if missing_prototype_cols:
        st.error(f"Prototype file is missing: {sorted(missing_prototype_cols)}")
        st.stop()

    if "type" not in prototypes.columns:
        prototypes["type"] = "condition"
    else:
        prototypes["type"] = prototypes["type"].fillna("condition").astype(str)
    condition_cols = prototypes.loc[
        prototypes["type"].str.lower().eq("condition"), "condition_name"
    ].astype(str).tolist()
    outcome_prototypes = prototypes.loc[
        prototypes["type"].str.lower().eq("outcome"), "condition_name"
    ].astype(str).tolist()
    if not condition_cols:
        st.error("At least one prototype must have type='condition'.")
        st.stop()

    try:
        scores = score_texts_against_prototypes(
            texts,
            prototypes,
            case_col=case_col,
            text_col=text_col,
            ngram_range=(min_ngram, max_ngram),
            keyword_weight=keyword_weight,
        )
        scores = add_low_signal_floor(scores)
        score_wide = wide_score_table(scores, value_col="calibration_score")
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    calibration_cols = condition_cols + (
        outcome_prototypes if outcome_choice == "Use prototype outcome" else []
    )
    anchors, thresholds = collect_calibration_controls(
        score_wide, calibration_cols, calibration_method
    )
    try:
        calibrated, calibration_rules = calibrate_scores(
            score_wide,
            condition_cols=calibration_cols,
            method=calibration_method,
            anchors=anchors,
            crisp_thresholds=thresholds,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    # --- Innovation 4: human review of near-threshold cases ---------------
    near_threshold = near_threshold_cases(
        score_wide, calibration_rules, condition_cols, band_fraction=review_band
    )
    review_table = pd.DataFrame()
    if not near_threshold.empty:
        review_table = near_threshold.merge(
            calibrated.melt(
                id_vars="case_id",
                value_vars=condition_cols,
                var_name="condition_name",
                value_name="calibrated_value",
            ),
            on=["case_id", "condition_name"],
            how="left",
        )
        review_table = review_table.merge(
            texts[[case_col, text_col]].rename(
                columns={case_col: "case_id", text_col: "text"}
            ),
            on="case_id",
            how="left",
        )
        overrides = st.session_state["human_overrides"]
        review_table["final_value"] = review_table.apply(
            lambda row: overrides.get((row["case_id"], row["condition_name"]), row["calibrated_value"]),
            axis=1,
        )
        review_table = review_table[
            ["case_id", "condition_name", "text", "raw_score", "anchor", "calibrated_value", "final_value"]
        ]

    calibrated_reviewed = apply_human_overrides(calibrated, st.session_state["human_overrides"])

    if outcome_choice == "Use prototype outcome":
        if not outcome_prototypes:
            st.error("No outcome column selected and no prototype has type='outcome'.")
            st.stop()
        outcome_col = outcome_prototypes[0]
        qca_ready = calibrated_reviewed[["case_id"] + condition_cols + [outcome_col]].rename(
            columns={outcome_col: "outcome"}
        )
        qca_outcome_col = "outcome"
    else:
        outcome_col = outcome_choice
        qca_ready = calibrated_reviewed[["case_id"] + condition_cols].merge(
            texts[[case_col, outcome_col]].rename(
                columns={case_col: "case_id", outcome_col: "outcome"}
            ),
            on="case_id",
            how="left",
        )
        qca_ready["outcome"] = coerce_outcome(qca_ready["outcome"])
        qca_outcome_col = "outcome"

    table = truth_table(
        qca_ready,
        condition_cols=condition_cols,
        outcome_col=qca_outcome_col,
        case_col="case_id",
        binarize_threshold=binarize_threshold,
        consistency_cutoff=consistency_cutoff,
        min_cases=int(min_cases),
    )
    solutions = solution_configurations(
        table,
        consistency_cutoff=consistency_cutoff,
        min_cases=int(min_cases),
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Cases", len(texts))
    metric_cols[1].metric("Conditions", len(condition_cols))
    metric_cols[2].metric("Truth rows", len(table))
    metric_cols[3].metric("Solutions", len(solutions))
    metric_cols[4].metric("Human overrides", len(st.session_state["human_overrides"]))

    (
        tab_scores,
        tab_calibration,
        tab_playground,
        tab_review,
        tab_qca,
        tab_truth,
        tab_solutions,
        tab_sensitivity,
        tab_figure,
    ) = st.tabs(
        [
            "📝 Scores",
            "🎚️ Calibration",
            "🎛️ Playground",
            "🔍 Human review",
            "📊 QCA dataset",
            "📋 Truth table",
            "✅ Solutions",
            "📈 Sensitivity",
            "🖼️ Figure",
        ]
    )

    with tab_scores:
        st.dataframe(scores, width="stretch", hide_index=True)
        download_button("Download score table", scores, "score_table.csv")
        download_button(
            "Download wide score table",
            score_wide,
            "classification_score_wide.csv",
        )

        st.subheader("Explain a score")
        st.caption(
            "Pick a case and a condition to see which words in the raw text matched "
            "the prototype's keywords, and how the n-gram and keyword components "
            "combined into the final score."
        )
        case_ids = scores["case_id"].unique().tolist()
        explain_case = st.selectbox("Case", case_ids, key="explain_case")
        explain_condition = st.selectbox(
            "Condition", scores["condition_name"].unique().tolist(), key="explain_condition"
        )
        match = scores[
            (scores["case_id"] == explain_case) & (scores["condition_name"] == explain_condition)
        ]
        if not match.empty:
            row = match.iloc[0]
            text_series = texts.loc[texts[case_col] == explain_case, text_col]
            text_value = text_series.iloc[0] if not text_series.empty else ""
            highlighted = highlight_matches(text_value, row["matched_keywords"])
            st.markdown(
                f"<div style='padding:10px;border:1px solid #d0d7de;border-radius:6px;"
                f"line-height:1.6;'>{highlighted}</div>",
                unsafe_allow_html=True,
            )
            st.caption(explain_score(row, keyword_weight))

    with tab_calibration:
        st.dataframe(calibration_rules, width="stretch", hide_index=True)
        st.dataframe(calibrated, width="stretch", hide_index=True)
        download_button("Download calibration rules", calibration_rules, "calibration_rules.csv")
        download_button(
            "Download calibrated membership",
            calibrated,
            "calibrated_membership.csv",
        )

    with tab_playground:
        st.caption(
            "Each histogram shows the raw prototype-similarity scores for one condition "
            "with the calibration anchor(s) overlaid. Adjust the anchor sliders in the "
            "sidebar to see how the membership split would change before committing to it."
        )
        rule_lookup = calibration_rules.set_index("condition_name")
        columns = st.columns(2)
        for index, condition in enumerate(condition_cols):
            target = columns[index % 2]
            values = pd.to_numeric(score_wide[condition], errors="coerce").dropna()
            rule = rule_lookup.loc[condition] if condition in rule_lookup.index else None
            with target:
                if px is not None and not values.empty:
                    fig = px.histogram(values, nbins=12, title=condition)
                    fig.update_traces(marker_color=COLOR_BLUE, marker_line_width=0)
                    fig.update_xaxes(title_text="raw score")
                    fig.update_yaxes(title_text="cases")
                    if rule is not None:
                        if calibration_method == "fuzzy" and pd.notna(rule.get("crossover")):
                            fig.add_vline(x=float(rule["full_out"]), line_dash="dot", line_color=COLOR_MUTED)
                            fig.add_vline(
                                x=float(rule["crossover"]),
                                line_dash="dash",
                                line_color=COLOR_ORANGE,
                                annotation_text="crossover",
                            )
                            fig.add_vline(x=float(rule["full_in"]), line_dash="dot", line_color=COLOR_MUTED)
                        elif pd.notna(rule.get("threshold")):
                            fig.add_vline(
                                x=float(rule["threshold"]),
                                line_dash="dash",
                                line_color=COLOR_ORANGE,
                                annotation_text="threshold",
                            )
                    fig.update_layout(
                        height=260,
                        margin=dict(t=40, b=20, l=20, r=20),
                        showlegend=False,
                        plot_bgcolor="#fcfcfb",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.bar_chart(values)

        st.subheader("Resulting membership buckets")
        st.caption("How the current calibration (including any human review overrides) splits cases per condition.")
        bucket_rows = []
        for condition in condition_cols:
            values = calibrated_reviewed[condition]
            bucket_rows.append(
                {
                    "condition_name": condition,
                    "low (<0.33)": int((values < 0.33).sum()),
                    "mid (0.33-0.66)": int(((values >= 0.33) & (values <= 0.66)).sum()),
                    "high (>0.66)": int((values > 0.66).sum()),
                }
            )
        st.dataframe(pd.DataFrame(bucket_rows), width="stretch", hide_index=True)

    with tab_review:
        st.caption(
            "Cases whose raw score falls close to the calibration anchor are listed below. "
            "A small change in wording (or in the anchor) could flip these from 'out' to "
            "'in', so they are good candidates for a researcher to check by hand instead of "
            "trusting the automatic score. Edit 'final_value' to override; overrides feed "
            "directly into the QCA-ready dataset, truth table, and downstream results."
        )
        if review_table.empty:
            st.info("No cases fall within the current ambiguity band.")
        else:
            edited = st.data_editor(
                review_table,
                width="stretch",
                hide_index=True,
                disabled=["case_id", "condition_name", "text", "raw_score", "anchor", "calibrated_value"],
                column_config={
                    "final_value": st.column_config.NumberColumn(
                        "final_value (editable)", min_value=0.0, max_value=1.0, step=0.05
                    )
                },
                key="human_review_editor",
            )
            changed = False
            for _, row in edited.iterrows():
                key = (row["case_id"], row["condition_name"])
                if abs(float(row["final_value"]) - float(row["calibrated_value"])) > 1e-9:
                    if st.session_state["human_overrides"].get(key) != float(row["final_value"]):
                        st.session_state["human_overrides"][key] = float(row["final_value"])
                        changed = True
                elif key in st.session_state["human_overrides"]:
                    del st.session_state["human_overrides"][key]
                    changed = True
            if changed:
                st.rerun()

            review_log = edited.rename(columns={"calibrated_value": "original_calibrated_value"})
            download_button("Download human review log", review_log, "human_review_log.csv")

    with tab_qca:
        st.dataframe(qca_ready, width="stretch", hide_index=True)
        download_button("Download QCA-ready dataset", qca_ready, "qca_ready_dataset.csv")

    with tab_truth:
        st.dataframe(table, width="stretch", hide_index=True)
        download_button("Download truth table", table, "truth_table.csv")

    with tab_solutions:
        if solutions.empty:
            st.warning("No configuration meets the current consistency and case thresholds.")
        else:
            st.dataframe(solutions, width="stretch", hide_index=True)
        download_button(
            "Download solution configurations",
            solutions,
            "solution_configurations.csv",
        )

    with tab_sensitivity:
        st.caption(
            "QCA results depend on two researcher choices: the consistency cutoff used to "
            "call a configuration 'sufficient', and the crossover used to binarize fuzzy "
            "membership into configurations. These sweeps hold everything else fixed and "
            "show how the solution count and average consistency/coverage move as each "
            "choice changes, so a single reported result isn't mistaken for the only one."
        )
        left, right = st.columns(2)
        with left:
            st.markdown("**Consistency cutoff sweep**")
            cutoff_lo, cutoff_hi = st.slider(
                "Cutoff range", 0.5, 1.0, (0.5, 0.95), 0.05, key="cutoff_sweep_range"
            )
            cutoffs = [round(float(v), 2) for v in np.arange(cutoff_lo, cutoff_hi + 0.001, 0.05)]
            cutoff_sweep = consistency_cutoff_sweep(
                qca_ready,
                condition_cols,
                qca_outcome_col,
                case_col="case_id",
                binarize_threshold=binarize_threshold,
                cutoffs=cutoffs,
                min_cases=int(min_cases),
            )
            if px is not None and not cutoff_sweep.empty:
                fig = px.line(cutoff_sweep, x="consistency_cutoff", y="n_solutions", markers=True)
                fig.update_traces(line_color=COLOR_BLUE, marker_color=COLOR_BLUE)
                fig.add_vline(x=consistency_cutoff, line_dash="dash", line_color=COLOR_MUTED)
                fig.update_layout(
                    height=280,
                    margin=dict(t=20, b=20, l=20, r=20),
                    plot_bgcolor="#fcfcfb",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, width="stretch")
            st.dataframe(cutoff_sweep, width="stretch", hide_index=True)
            download_button("Download cutoff sweep", cutoff_sweep, "sensitivity_cutoff_sweep.csv")

        with right:
            st.markdown("**Configuration crossover sweep**")
            cross_lo, cross_hi = st.slider(
                "Crossover range", 0.2, 0.8, (0.3, 0.7), 0.05, key="threshold_sweep_range"
            )
            cross_thresholds = [round(float(v), 2) for v in np.arange(cross_lo, cross_hi + 0.001, 0.05)]
            threshold_sweep = binarize_threshold_sweep(
                qca_ready,
                condition_cols,
                qca_outcome_col,
                case_col="case_id",
                consistency_cutoff=consistency_cutoff,
                thresholds=cross_thresholds,
                min_cases=int(min_cases),
            )
            if px is not None and not threshold_sweep.empty:
                fig2 = px.line(threshold_sweep, x="binarize_threshold", y="n_solutions", markers=True)
                fig2.update_traces(line_color=COLOR_BLUE, marker_color=COLOR_BLUE)
                fig2.add_vline(x=binarize_threshold, line_dash="dash", line_color=COLOR_MUTED)
                fig2.update_layout(
                    height=280,
                    margin=dict(t=20, b=20, l=20, r=20),
                    plot_bgcolor="#fcfcfb",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig2, width="stretch")
            st.dataframe(threshold_sweep, width="stretch", hide_index=True)
            download_button("Download crossover sweep", threshold_sweep, "sensitivity_threshold_sweep.csv")

    with tab_figure:
        heatmap_data = qca_ready.set_index("case_id")[condition_cols + ["outcome"]]
        if px is not None:
            fig = px.imshow(
                heatmap_data,
                color_continuous_scale=BLUE_SEQUENTIAL_SCALE,
                zmin=0,
                zmax=1,
                aspect="auto",
                labels=dict(x="Condition / outcome", y="Case", color="Membership"),
            )
            fig.update_layout(height=max(420, 24 * len(heatmap_data)), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.dataframe(heatmap_data.style.background_gradient(cmap="Blues"))

        if px is not None and not table.empty:
            scatter = px.scatter(
                table,
                x="consistency",
                y="coverage",
                size="n_cases",
                color="status",
                color_discrete_map=STATUS_COLORS,
                hover_data=["configuration", "cases"],
                range_x=[0, 1.02],
                range_y=[0, 1.02],
            )
            scatter.add_vline(x=consistency_cutoff, line_dash="dash", line_color=COLOR_MUTED)
            scatter.update_layout(plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(scatter, width="stretch")

    with st.expander("Method note", expanded=False):
        st.markdown(
            """
Scores combine character n-gram cosine similarity and optional keyword overlap.
Fuzzy calibration uses default 25th percentile, median, and 75th percentile anchors
unless overridden in the sidebar. Configuration consistency is calculated as
`sum(min(X, Y)) / sum(X)` and coverage as `sum(min(X, Y)) / sum(Y)`. Cases flagged
in the Human review tab and given a manual override are applied before the
QCA-ready dataset, truth table, and all downstream results are built.
            """.strip()
        )


if __name__ == "__main__":
    main()
