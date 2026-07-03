# Technical Note

## What the Tool Does

This tool converts raw text data into QCA-ready conditions and basic QCA results. It is designed for applied social science research settings where a researcher has citizen messages, public consultation comments, policy texts, or government replies and wants to turn conceptual categories into transparent set-membership values.

The workflow has five stages. First, the user uploads a text dataset and a prototype file. Second, each text is scored against each conceptual prototype. Third, the raw scores are calibrated into fuzzy-set or crisp-set memberships. Fourth, the calibrated memberships are combined with an outcome to produce a QCA-ready dataset. Fifth, the tool generates a truth table, consistency and coverage metrics, solution configurations, and interpretation-oriented figures.

## Required Data

The text dataset should contain one row per case, a case identifier, a text column, and preferably an outcome column. The outcome can be crisp, such as 0 or 1, or fuzzy, such as a value between 0 and 1. If the dataset does not contain an outcome column, the user can provide an outcome prototype and the tool can calibrate that prototype score as the outcome.

The prototype file should contain a `condition_name`, a `prototype` description, a `type`, and optional `keywords`. The `type` column identifies whether the prototype is a causal condition or an outcome. Both the `prototype` text and the `keywords` are free text and can be written in whatever language the source texts are in. The included demo prototype file is bilingual: each `prototype` sentence pairs an English clause with a Chinese paraphrase, and `keywords` lists matching English and Chinese terms, so every string the scorer uses is visible in the uploaded CSV rather than hidden in code.

## Scoring Method

The tool uses a hybrid prototype-based scoring method. It calculates character n-gram cosine similarity between each text and each prototype. Character n-grams are useful for Chinese because they do not require word segmentation. If a prototype also contains keywords, the tool calculates the share of keywords that appear in the text. The final score is a weighted combination of n-gram similarity and keyword overlap. The default keyword weight is 0.35. Before calibration, the tool applies a low-signal floor: if there is no keyword evidence for a condition and the raw score is below 0.01, the calibration input is set to zero while the original raw score is retained in the score table. This prevents tiny surface-overlap noise from being amplified by percentile calibration in a small demo dataset.

This approach is intentionally transparent. The score table reports the n-gram score, keyword score, matched keywords, and final score. A researcher can inspect why a text received a high or low score and revise the prototype or keyword list if necessary.

## Calibration Choices

The tool supports both fuzzy-set and crisp-set calibration. For fuzzy sets, the default anchors are the 25th percentile, median, and 75th percentile of the raw scores for each condition. These correspond to full non-membership, crossover, and full membership. For crisp sets, the default threshold is the median score. The Streamlit interface exposes these anchors and thresholds in the sidebar so the researcher can adjust them when theoretical or substantive anchors are available.

## QCA Outputs

The QCA-ready dataset has one row per case, one column per calibrated condition, and one outcome column. The truth table shows observed configurations, number of cases, positive and negative cases, consistency, coverage, outcome value, status, and the case ids in each configuration. Consistency is calculated as `sum(min(X, Y)) / sum(X)`. Coverage is calculated as `sum(min(X, Y)) / sum(Y)`. Configurations that meet the selected consistency cutoff and minimum case count are reported as solution configurations.

## Research-Design Features Beyond the Minimum Workflow

Four features go beyond upload-score-calibrate-solve, each addressing a specific research-design concern rather than adding generic polish:

- **Explainable scoring.** For any case/condition pair, the app shows the raw text with the matched keyword spans highlighted and the score formula with actual numbers filled in (e.g. `score = 0.65*ngram(0.00) + 0.35*keyword(0.07)`). This lets a reviewer check *why* a score is what it is without reading source code.
- **Calibration playground.** A per-condition histogram of raw scores with the fuzzy anchors (or crisp threshold) drawn as vertical lines, plus a count of how many cases fall into low/mid/high membership after calibration. Moving an anchor slider and checking this view shows the consequence of that choice before it is locked into the QCA-ready dataset.
- **Human review of near-threshold cases.** Cases whose raw score sits within an adjustable band of the calibration anchor are flagged as ambiguous and listed with their original text in an editable table. A researcher can override the calibrated value for any flagged case; overrides are applied before the QCA-ready dataset and every downstream table are built, and are exportable as a review log. This is a deliberate concession that automated scoring should not make the final call on borderline cases.
- **Threshold sensitivity analysis.** Two sweeps — over the consistency cutoff and over the configuration crossover — report how the number of solution configurations and their average consistency/coverage change, holding everything else fixed. Because raising the cutoff can only shrink or hold the solution set, this makes visible whether a reported solution is a robust pattern or an artifact of one particular cutoff choice.

All four features read from and write into the same `qca_text_tool.scoring` / `qca_text_tool.calibration` / `qca_text_tool.qca` functions used by the core workflow; they do not introduce a second, parallel scoring or QCA implementation.

## Interpretation and Limitations

The results should be interpreted as a structured aid for research judgment. A high score means the text is closer to a prototype according to the transparent scoring rule; it does not prove that the concept is truly present. Prototype quality, keyword selection, ambiguous language, sarcasm, mixed topics, and short texts can affect the results. Calibration is also consequential: different anchors can produce different truth tables and solution configurations.

The scoring method is a character n-gram and keyword-overlap similarity, not a semantic embedding model, so it can miss paraphrases that share no surface form with the prototype text or keywords. Because the demo texts are Chinese and n-gram overlap only exists where the prototype text itself contains Chinese characters, the quality of the Chinese half of each `prototype` sentence and the Chinese `keywords` entries directly drives the demo scores; a poorly translated prototype will under-score genuinely relevant text. The current tool also reports observed sufficient configurations rather than a full minimized parsimonious or intermediate QCA solution, and the sensitivity sweeps vary one threshold at a time rather than jointly. The human review step also does not persist across sessions or scale past a small, manually inspectable set of flagged cases. With more time, I would add optional multilingual embedding models, a full Quine-McCluskey-style minimization procedure, joint (not one-at-a-time) threshold sensitivity, and a way to save/reload human review decisions for larger datasets.
