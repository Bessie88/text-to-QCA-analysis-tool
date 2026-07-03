# Text-to-QCA Analysis Tool

This project is a take-home assessment submission for Task 2: converting raw text into QCA-ready conditions and basic QCA results.

The tool lets a researcher upload a text dataset and conceptual prototypes, score each text against the prototypes, calibrate those scores into set memberships, build a QCA-ready dataset, generate a truth table, calculate consistency and coverage, and export the main outputs.

## Quick Start

```bash
cd task2_text_qca_tool
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app loads the included demo files by default:

- `data/demo_texts.csv`
- `data/prototypes.csv`

## Input Format

The text dataset must be a CSV file with one row per case. It should include:

- a case id column, such as `case_id`
- a text column, such as `text`
- an optional outcome column, such as `outcome`

Example:

```csv
case_id,text,outcome
1,sample citizen message,1
2,another sample citizen message,0
```

The prototype file must include:

- `condition_name`
- `prototype`
- `type`, either `condition` or `outcome`
- optional `keywords`

Example:

```csv
condition_name,prototype,type,keywords
dissatisfaction,"The citizen expresses dissatisfaction, anger, complaint, frustration, or distrust.",condition,"dissatisfaction;anger;complaint;frustration;distrust"
```

For multilingual text, write the `prototype` description and the `keywords` in the same language as the texts (or mix languages freely). The included demo prototype file is bilingual: each `prototype` sentence has an English clause plus a Chinese paraphrase, and `keywords` lists both English and Chinese terms, so the scoring uses only what is visible in the CSV — there is no hidden per-language lookup table in the code.

## Method

The default scorer combines:

1. character n-gram cosine similarity between each text and each prototype
2. optional keyword overlap using the `keywords` column

The final raw score is:

```text
score = (1 - keyword_weight) * ngram_score + keyword_weight * keyword_score
```

If a prototype has no keywords, the score is just the n-gram similarity.

Before calibration, the tool applies a low-signal floor: if a case has no
matched keywords for a condition and the raw score is below `0.01`, its
`calibration_score` is set to `0`. The original `score` is still preserved in
`score_table.csv`; this guard prevents tiny character-overlap noise from being
amplified by percentile-based calibration in small demo datasets.

The calibration step supports:

- fuzzy-set membership
- crisp-set membership

For fuzzy sets, the default anchors are:

- full out: 25th percentile of raw scores
- crossover: median raw score
- full in: 75th percentile of raw scores

For crisp sets, the default threshold is the median raw score.

The Streamlit sidebar lets users inspect and adjust these anchors or thresholds
before generating the final QCA table.

The QCA metrics use standard formulas:

```text
consistency(X -> Y) = sum(min(X, Y)) / sum(X)
coverage(X -> Y) = sum(min(X, Y)) / sum(Y)
```

Observed configurations that meet the selected consistency cutoff and minimum case count are reported as solution configurations.

## Beyond the Core Workflow

The Streamlit app adds four features on top of the minimum upload-to-solution pipeline:

- **Explainable scoring** (Scores tab). Pick any case and condition to see the raw text with matched keyword spans highlighted, plus a one-line breakdown of the score formula (`score = (1-w)*ngram + w*keyword`) with the actual numbers substituted in. This is meant to answer "why did this text get this score" without reading code.
- **Calibration playground** (Calibration playground tab). A histogram of raw scores per condition with the fuzzy anchors (or crisp threshold) drawn as vertical lines, plus a table of how many cases fall into low/mid/high membership buckets. Moving the anchor sliders in the sidebar and revisiting this tab shows how the split would change before committing to it.
- **Human review of near-threshold cases** (Human review tab). Any case whose raw score sits within an adjustable band of the calibration anchor is flagged as ambiguous — a small wording difference could flip it. These cases are listed with their original text and are editable in place; overrides are applied before the QCA-ready dataset, truth table, and every downstream result are built, and can be exported as a review log. This keeps automated scoring from silently overriding a researcher's judgment on borderline cases.
- **Threshold sensitivity analysis** (Sensitivity tab). Sweeps the consistency cutoff and, separately, the configuration crossover, holding everything else fixed, and reports how the number of solution configurations and their average consistency/coverage move. Since raising the cutoff can only shrink or hold the solution set, this makes visible how fragile (or robust) the reported solution is to that one choice, instead of presenting a single cutoff's result as definitive.

None of these change the core formulas in `qca_text_tool/`; they are transparency and robustness layers around the same scoring, calibration, and QCA logic.

## Sample Outputs

Run:

```bash
python scripts/generate_sample_outputs.py
```

This creates:

- `outputs/score_table.csv`
- `outputs/classification_score_wide.csv`
- `outputs/calibrated_membership.csv`
- `outputs/calibration_rules.csv`
- `outputs/qca_ready_dataset.csv`
- `outputs/truth_table.csv`
- `outputs/solution_configurations.csv`
- `outputs/human_review_candidates.csv` — cases flagged as near the calibration anchor, with their raw text (no manual overrides applied; this is what the Human review tab shows before any edits)
- `outputs/sensitivity_cutoff_sweep.csv` — solution count and average consistency/coverage across a range of consistency cutoffs
- `outputs/sensitivity_threshold_sweep.csv` — number of configurations and solutions across a range of configuration crossovers
- `outputs/membership_heatmap.html`

## Reproducibility

The core algorithm is deterministic. The local scorer does not call an external API and does not require model downloads. All demo outputs can be regenerated from the included demo data and prototype file.

## Limitations

This tool is designed as a transparent research prototype, not as a fully automated substitute for qualitative judgment. Prototype wording and keyword quality strongly affect scores. Calibration choices can change QCA results, especially with small samples. The solution output reports observed sufficient configurations rather than a full minimized parsimonious or intermediate QCA solution.
