"""Prototype-based text scoring using multilingual sentence embeddings.

The default scorer combines semantic cosine similarity (from a multilingual
sentence-transformers model) between each text and each prototype description
with optional keyword overlap. Both the prototype text and the keywords
column are free-form and may be written in any language (including mixed
English/Chinese); the embedding model itself is multilingual and supports
cross-lingual matching (e.g. a Chinese text against an English prototype).
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd


DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
KEYWORD_SPLIT_RE = re.compile(r"[,;|，、\n]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
CJK_NEGATION_RE = re.compile(r"(不|没|没有|未|无|非|缺乏|难以)$")
CJK_NEGATION_PREFIXES = ("不", "没", "没有", "未", "无", "非", "缺乏", "难以")


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


@lru_cache(maxsize=None)
def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _semantic_similarity_matrix(
    texts: list[str], prototypes: list[str], model_name: str
) -> np.ndarray:
    """Return a (len(texts) x len(prototypes)) cosine similarity matrix."""
    model = _load_embedding_model(model_name)
    embeddings = model.encode(
        [*texts, *prototypes],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    text_embeddings = embeddings[: len(texts)]
    prototype_embeddings = embeddings[len(texts) :]
    return text_embeddings @ prototype_embeddings.T


def _cjk_keyword_matches(raw: str, keyword: str) -> bool:
    """Return True for CJK keyword hits, avoiding simple negated false positives."""
    keyword = keyword.lower().strip()
    if not keyword:
        return False
    if keyword.startswith(CJK_NEGATION_PREFIXES):
        return keyword in raw

    start = 0
    while True:
        index = raw.find(keyword, start)
        if index == -1:
            return False
        prefix = raw[max(0, index - 3) : index]
        if not CJK_NEGATION_RE.search(prefix):
            return True
        start = index + len(keyword)


def _keyword_in_text(raw: str, clean: str, keyword: str) -> bool:
    normalized_keyword = clean_text(keyword)
    if not normalized_keyword:
        return False
    if CJK_RE.search(keyword):
        return _cjk_keyword_matches(raw, keyword)

    tokens = normalized_keyword.split()
    if not tokens:
        return False
    pattern = re.compile(
        r"(?<![A-Za-z0-9])"
        + r"\s+".join(map(re.escape, tokens))
        + r"(?![A-Za-z0-9])"
    )
    return pattern.search(clean) is not None


def _keyword_match_score(text: str, keywords: tuple[str, ...]) -> tuple[float, str]:
    if not keywords:
        return np.nan, ""
    clean = clean_text(text)
    raw = str(text).lower()
    matched = []
    for keyword in keywords:
        if _keyword_in_text(raw, clean, keyword):
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
    model_name: str = DEFAULT_MODEL_NAME,
    keyword_weight: float = 0.35,
) -> pd.DataFrame:
    """Return long-form prototype scores for every case/prototype pair."""
    if case_col not in texts_df.columns:
        raise ValueError(f"Case id column '{case_col}' is not in the text dataset.")
    if text_col not in texts_df.columns:
        raise ValueError(f"Text column '{text_col}' is not in the text dataset.")

    keyword_weight = min(max(float(keyword_weight), 0.0), 1.0)
    prototypes = _load_prototypes(prototypes_df)

    text_values = texts_df[text_col].fillna("").astype(str).tolist()
    prototype_values = [prototype.prototype for prototype in prototypes]
    similarity_matrix = _semantic_similarity_matrix(text_values, prototype_values, model_name)

    rows = []
    for text_idx, (_, text_row) in enumerate(texts_df.iterrows()):
        case_id = text_row[case_col]
        text = text_row[text_col]
        for prototype_idx, prototype in enumerate(prototypes):
            # Cosine similarity between embeddings can be slightly negative for
            # unrelated text; clip to 0 since membership scores are in [0, 1].
            semantic_score = max(0.0, float(similarity_matrix[text_idx, prototype_idx]))
            keyword_score, matched_keywords = _keyword_match_score(text, prototype.keywords)
            if math.isnan(keyword_score):
                score = semantic_score
            else:
                score = (1.0 - keyword_weight) * semantic_score + keyword_weight * keyword_score
            rows.append(
                {
                    "case_id": case_id,
                    "condition_name": prototype.condition_name,
                    "prototype_type": prototype.prototype_type,
                    "semantic_score": round(semantic_score, 6),
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
    semantic_score = float(row["semantic_score"])
    keyword_score = row.get("keyword_score")
    if keyword_score is None or (isinstance(keyword_score, float) and math.isnan(keyword_score)):
        return f"score = semantic_similarity ({semantic_score:.3f}); no keywords supplied for this condition"
    keyword_score = float(keyword_score)
    matched = row.get("matched_keywords") or "none"
    return (
        f"score = (1 - {keyword_weight:.2f}) x semantic ({semantic_score:.3f}) "
        f"+ {keyword_weight:.2f} x keyword ({keyword_score:.3f}); matched: {matched}"
    )


def add_low_signal_floor(
    score_table: pd.DataFrame,
    score_floor: float = 0.01,
    source_col: str = "score",
    output_col: str = "calibration_score",
) -> pd.DataFrame:
    """Add a calibration score that suppresses low, unmatched surface overlap.

    Percentile calibration can exaggerate tiny low-signal similarities when a demo set is
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
