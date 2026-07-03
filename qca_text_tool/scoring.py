"""Prototype-based text scoring without external model downloads.

The default scorer combines character n-gram cosine similarity between each
text and each prototype description with optional keyword overlap. Both the
prototype text and the keywords column are free-form and may be written in
any language (including mixed English/Chinese), which is what drives the
score for non-English text -- there is no hidden per-language lookup table.
"""

from __future__ import annotations

import html
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
KEYWORD_SPLIT_RE = re.compile(r"[,;|，、\n]+")


@dataclass(frozen=True)
class Prototype:
    condition_name: str
    prototype: str
    prototype_type: str
    keywords: tuple[str, ...]


def clean_text(value: object) -> str:
    """Return normalized text while preserving Chinese characters."""
    if pd.isna(value):
        return ""
    text = str(value).lower()
    tokens = TOKEN_RE.findall(text)
    return " ".join(tokens)


def parse_keywords(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return tuple()
    keywords = []
    for item in KEYWORD_SPLIT_RE.split(str(value)):
        item = item.strip().lower()
        if item:
            keywords.append(item)
    return tuple(dict.fromkeys(keywords))


def _char_ngrams(text: str, ngram_range: tuple[int, int]) -> Counter[str]:
    min_n, max_n = ngram_range
    grams: Counter[str] = Counter()
    for token in TOKEN_RE.findall(clean_text(text)):
        if not token:
            continue
        for n in range(min_n, max_n + 1):
            if len(token) < n:
                if len(token) >= min_n:
                    grams[token] += 1
                continue
            for idx in range(0, len(token) - n + 1):
                grams[token[idx : idx + n]] += 1
    return grams


def _tfidf_vectors(
    documents: Iterable[str], ngram_range: tuple[int, int]
) -> list[dict[str, float]]:
    counts = [_char_ngrams(doc, ngram_range) for doc in documents]
    doc_count = len(counts)
    document_frequency: Counter[str] = Counter()
    for doc_counts in counts:
        document_frequency.update(doc_counts.keys())

    vectors: list[dict[str, float]] = []
    for doc_counts in counts:
        vector: dict[str, float] = {}
        for gram, count in doc_counts.items():
            tf = 1.0 + math.log(count)
            idf = math.log((1.0 + doc_count) / (1.0 + document_frequency[gram])) + 1.0
            vector[gram] = tf * idf
        vectors.append(vector)
    return vectors


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    numerator = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))


def _keyword_match_score(text: str, keywords: tuple[str, ...]) -> tuple[float, str]:
    if not keywords:
        return np.nan, ""
    clean = clean_text(text)
    raw = str(text).lower()
    matched = []
    for keyword in keywords:
        normalized_keyword = clean_text(keyword)
        if not normalized_keyword:
            continue
        if normalized_keyword in clean or keyword.lower() in raw:
            matched.append(keyword)
    return len(matched) / len(keywords), "; ".join(dict.fromkeys(matched))


def _load_prototypes(prototypes_df: pd.DataFrame) -> list[Prototype]:
    required = {"condition_name", "prototype"}
    missing = required.difference(prototypes_df.columns)
    if missing:
        raise ValueError(f"Prototype file is missing required columns: {sorted(missing)}")

    prototypes: list[Prototype] = []
    for _, row in prototypes_df.iterrows():
        name = str(row["condition_name"]).strip()
        if not name:
            continue
        prototype_type = str(row.get("type", "condition")).strip().lower() or "condition"
        keywords = parse_keywords(row.get("keywords", ""))
        prototypes.append(
            Prototype(
                condition_name=name,
                prototype=str(row["prototype"]),
                prototype_type=prototype_type,
                keywords=keywords,
            )
        )
    if not prototypes:
        raise ValueError("Prototype file does not contain any usable prototypes.")
    return prototypes


def score_texts_against_prototypes(
    texts_df: pd.DataFrame,
    prototypes_df: pd.DataFrame,
    case_col: str,
    text_col: str,
    ngram_range: tuple[int, int] = (2, 4),
    keyword_weight: float = 0.35,
) -> pd.DataFrame:
    """Return long-form prototype scores for every case/prototype pair."""
    if case_col not in texts_df.columns:
        raise ValueError(f"Case id column '{case_col}' is not in the text dataset.")
    if text_col not in texts_df.columns:
        raise ValueError(f"Text column '{text_col}' is not in the text dataset.")
    if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
        raise ValueError("ngram_range must be a valid (min_n, max_n) pair.")

    keyword_weight = min(max(float(keyword_weight), 0.0), 1.0)
    prototypes = _load_prototypes(prototypes_df)

    text_values = texts_df[text_col].fillna("").astype(str).tolist()
    prototype_values = [prototype.prototype for prototype in prototypes]
    vectors = _tfidf_vectors(text_values + prototype_values, ngram_range)
    text_vectors = vectors[: len(text_values)]
    prototype_vectors = vectors[len(text_values) :]

    rows = []
    for text_idx, (_, text_row) in enumerate(texts_df.iterrows()):
        case_id = text_row[case_col]
        text = text_row[text_col]
        for prototype, prototype_vector in zip(prototypes, prototype_vectors):
            ngram_score = _cosine_similarity(text_vectors[text_idx], prototype_vector)
            keyword_score, matched_keywords = _keyword_match_score(text, prototype.keywords)
            if math.isnan(keyword_score):
                score = ngram_score
            else:
                score = (1.0 - keyword_weight) * ngram_score + keyword_weight * keyword_score
            rows.append(
                {
                    "case_id": case_id,
                    "condition_name": prototype.condition_name,
                    "prototype_type": prototype.prototype_type,
                    "ngram_score": round(float(ngram_score), 6),
                    "keyword_score": None
                    if math.isnan(keyword_score)
                    else round(float(keyword_score), 6),
                    "matched_keywords": matched_keywords,
                    "score": round(float(score), 6),
                }
            )
    return pd.DataFrame(rows)


def highlight_matches(text: object, matched_keywords: object) -> str:
    """Return HTML with each matched keyword span wrapped in ``<mark>``.

    Used by the "explainable scoring" view so a researcher can see exactly
    which substring of the raw text drove the keyword-overlap component of
    a score, instead of trusting the number alone.
    """
    text = "" if pd.isna(text) else str(text)
    keywords = [] if pd.isna(matched_keywords) else str(matched_keywords).split("; ")
    keywords = sorted({keyword for keyword in keywords if keyword}, key=len, reverse=True)
    if not keywords:
        return html.escape(text)

    pattern = re.compile("|".join(re.escape(keyword) for keyword in keywords), re.IGNORECASE)
    pieces = []
    last_end = 0
    for match in pattern.finditer(text):
        pieces.append(html.escape(text[last_end : match.start()]))
        pieces.append(f"<mark>{html.escape(match.group(0))}</mark>")
        last_end = match.end()
    pieces.append(html.escape(text[last_end:]))
    return "".join(pieces)


def explain_score(row: pd.Series, keyword_weight: float) -> str:
    """Return a one-line, human-readable breakdown of how a score was built."""
    ngram_score = float(row["ngram_score"])
    keyword_score = row.get("keyword_score")
    if keyword_score is None or (isinstance(keyword_score, float) and math.isnan(keyword_score)):
        return f"score = ngram_similarity ({ngram_score:.3f}); no keywords supplied for this condition"
    keyword_score = float(keyword_score)
    matched = row.get("matched_keywords") or "none"
    return (
        f"score = (1 - {keyword_weight:.2f}) x ngram ({ngram_score:.3f}) "
        f"+ {keyword_weight:.2f} x keyword ({keyword_score:.3f}); matched: {matched}"
    )


def add_low_signal_floor(
    score_table: pd.DataFrame,
    score_floor: float = 0.01,
    source_col: str = "score",
    output_col: str = "calibration_score",
) -> pd.DataFrame:
    """Add a calibration score that suppresses low, unmatched surface overlap.

    Percentile calibration can exaggerate tiny n-gram overlaps when a demo set is
    small and many raw scores are near zero. If a case has no keyword evidence
    for a condition and the raw score is below ``score_floor``, the calibration
    input is set to zero while the original score is preserved for inspection.
    """
    required = {source_col, "matched_keywords"}
    missing = required.difference(score_table.columns)
    if missing:
        raise ValueError(f"Score table is missing required columns: {sorted(missing)}")

    result = score_table.copy()
    raw_scores = pd.to_numeric(result[source_col], errors="coerce").fillna(0.0)
    matched = result["matched_keywords"].fillna("").astype(str).str.strip()
    floor_mask = matched.eq("") & (raw_scores < float(score_floor))
    result[output_col] = raw_scores.where(~floor_mask, 0.0).round(6)
    result["low_signal_floor_applied"] = floor_mask
    return result


def wide_score_table(score_table: pd.DataFrame, value_col: str = "score") -> pd.DataFrame:
    """Pivot long scores into one row per case and one score column per prototype."""
    required = {"case_id", "condition_name", value_col}
    missing = required.difference(score_table.columns)
    if missing:
        raise ValueError(f"Score table is missing required columns: {sorted(missing)}")
    wide = score_table.pivot_table(
        index="case_id",
        columns="condition_name",
        values=value_col,
        aggfunc="mean",
    )
    wide = wide.reset_index()
    wide.columns.name = None
    return wide
