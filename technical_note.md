# Technical Note

## What the Tool Does

This tool converts raw text into QCA-ready conditions and basic QCA results. It is designed for applied social science research settings where a researcher has citizen messages, policy texts, public consultation comments, or government replies and wants to turn conceptual categories into transparent set-membership values. The workflow has five stages: upload text data and conceptual prototypes, score each text against each prototype, calibrate the scores into fuzzy-set or crisp-set memberships, combine calibrated conditions with an outcome, and generate a truth table, consistency and coverage metrics, solution configurations, and interpretation-oriented figures.

## Required Data

The text dataset should be a CSV file with one row per case, a case identifier, a text column, and preferably an outcome column. The outcome can be crisp, such as 0 or 1, or fuzzy, such as a value between 0 and 1. If the dataset does not contain an outcome column, the user can provide an outcome prototype and the tool can score and calibrate that prototype as the outcome.

The prototype file should contain `condition_name`, `prototype`, `type`, and optional `keywords`. The `type` column marks whether the prototype is a causal condition or an outcome. Prototype descriptions and keywords are free text and can be written in the same language as the source texts or in mixed languages. The included demo prototype file is bilingual: each prototype has an English description plus a Chinese paraphrase, and the keyword lists include both English and Chinese terms. This keeps the scoring rule visible in the uploaded CSV rather than hidden in code.

## Scoring Method

The tool uses a hybrid prototype-based scoring method. It calculates semantic cosine similarity between each text and each prototype using the multilingual sentence-transformer model `paraphrase-multilingual-MiniLM-L12-v2`. This supports cross-lingual matching, such as Chinese text scored against an English prototype, and captures paraphrases that do not share exact surface words. If keywords are provided, the tool also calculates the share of prototype keywords that appear in the text. The final score is a weighted combination of semantic similarity and keyword overlap, with a default keyword weight of 0.35.

Before calibration, the tool applies a low-signal floor: if a case has no keyword evidence for a condition and its raw score is below 0.01, the calibration input is set to zero while the original score is retained in the score table. This prevents tiny embedding-similarity noise from being amplified by percentile calibration in a small demo dataset.

This approach is intentionally inspectable. The score table reports semantic score, keyword score, matched keywords, final score, calibration score, and whether the low-signal floor was applied. In the app, a researcher can pick any case and condition to see matched keyword spans highlighted in the raw text and a one-line score formula with the actual numbers filled in.

## Calibration Choices

The tool supports both fuzzy-set and crisp-set calibration. For fuzzy sets, the default anchors are the 25th percentile, median, and 75th percentile of the raw scores for each condition. These correspond to full non-membership, crossover, and full membership. For crisp sets, the default threshold is the median score. The Streamlit sidebar exposes these anchors and thresholds so a researcher can adjust them when theory or substantive knowledge suggests better anchors.

The demo output uses one substantive override for the dissatisfaction condition. The default percentile crossover placed some cooperative or grateful cases too close to complaint cases because they share general citizen-government language. The demo therefore sets the dissatisfaction crossover at 0.22, which falls between the mildest genuine complaint and the strongest cooperative false-positive case. This illustrates that calibration is a research-design decision, not just a mechanical step.

## QCA Outputs

The QCA-ready dataset has one row per case, one column per calibrated condition, and one outcome column. The truth table shows observed configurations, number of cases, positive and negative cases, consistency, coverage, outcome value, status, and case ids. Consistency is calculated as `sum(min(X, Y)) / sum(X)`, and coverage is calculated as `sum(min(X, Y)) / sum(Y)`. Configurations that meet the selected consistency cutoff and minimum case count are reported as solution configurations. The tool also identifies weak, contradictory, and not-sufficient configurations.

The app exports the score table, wide score table, calibration rules, calibrated membership table, QCA-ready dataset, truth table, solution configurations, human review candidates, sensitivity sweeps, and a membership heatmap. The included script regenerates the same sample outputs from the demo data.

## Research-Design Features

Several features go beyond the minimum workflow. The calibration playground shows score distributions with the current anchors overlaid, plus membership bucket counts. The human review tab flags cases close to the calibration anchor and allows a researcher to override borderline memberships before the QCA-ready dataset and downstream results are built. The sensitivity tab sweeps the consistency cutoff and configuration crossover to show how solution counts and average consistency or coverage change. These features are meant to make researcher judgment visible rather than presenting one automated result as final.

## Interpretation and Limitations

The results should be interpreted as a structured aid for qualitative and configurational analysis, not as a fully automated substitute for coding judgment. A high score means the text is close to a prototype according to the stated scoring rule; it does not prove that the concept is truly present. Prototype wording, keyword quality, ambiguous language, sarcasm, mixed topics, and short texts can all affect scores. Calibration choices are also consequential, especially with small samples.

The multilingual embedding model improves cross-lingual and paraphrase matching, but it can still miss negation, sarcasm, or domain-specific meanings. The current tool reports observed sufficient configurations rather than a fully minimized parsimonious or intermediate QCA solution. Sensitivity checks vary one threshold at a time rather than jointly, and human review decisions are session-based rather than saved for large projects. With more time, I would benchmark alternative multilingual embedding models, add formal QCA minimization, support joint sensitivity analysis, and add persistent review logs for larger datasets.
