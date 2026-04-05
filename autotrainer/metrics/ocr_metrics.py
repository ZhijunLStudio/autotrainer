"""OCR-specific evaluation metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


def _levenshtein_distance(s: str, t: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s) < len(t):
        return _levenshtein_distance(t, s)
    if not t:
        return len(s)

    prev_row = list(range(len(t) + 1))
    for i, c1 in enumerate(s):
        curr_row = [i + 1]
        for j, c2 in enumerate(t):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def compute_ned(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Normalized Edit Distance. Lower is better. Range [0, 1]."""
    if not predictions:
        return 0.0
    total = 0.0
    for pred, ref in zip(predictions, references):
        dist = _levenshtein_distance(pred, ref)
        max_len = max(len(pred), len(ref))
        if max_len > 0:
            total += dist / max_len
    return total / len(predictions)


def compute_cer(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Character Error Rate. Lower is better."""
    total_dist = 0
    total_ref_chars = 0
    for pred, ref in zip(predictions, references):
        total_dist += _levenshtein_distance(pred, ref)
        total_ref_chars += len(ref)
    if total_ref_chars == 0:
        return 0.0
    return total_dist / total_ref_chars


def compute_wer(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Word Error Rate. Lower is better."""
    total_dist = 0
    total_ref_words = 0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        total_dist += _levenshtein_distance(pred_words, ref_words) if pred_words or ref_words else 0
        total_ref_words += len(ref_words)
    if total_ref_words == 0:
        return 0.0
    return total_dist / total_ref_words


def compute_exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Exact Match rate. Higher is better. Range [0, 1]."""
    if not predictions:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if p == r)
    return matches / len(predictions)


@dataclass
class OCRMetrics:
    """Container for all OCR text recognition metrics."""
    ned: float = 0.0
    cer: float = 0.0
    wer: float = 0.0
    exact_match: float = 0.0
    num_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "ned": round(self.ned, 6),
            "cer": round(self.cer, 6),
            "wer": round(self.wer, 6),
            "exact_match": round(self.exact_match, 6),
            "num_samples": self.num_samples,
        }


def compute_ocr_text_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
) -> OCRMetrics:
    """Compute all text recognition metrics at once."""
    return OCRMetrics(
        ned=compute_ned(predictions, references),
        cer=compute_cer(predictions, references),
        wer=compute_wer(predictions, references),
        exact_match=compute_exact_match(predictions, references),
        num_samples=len(predictions),
    )


def compute_teds(pred_html: str, ref_html: str) -> float:
    """Tree Edit Distance Similarity for table structure (simplified)."""
    def _extract_tags(html: str) -> list[str]:
        return re.findall(r"<(/?\w+)[^>]*>", html)

    pred_tags = _extract_tags(pred_html)
    ref_tags = _extract_tags(ref_html)
    if not ref_tags:
        return 1.0 if not pred_tags else 0.0

    dist = _levenshtein_distance(pred_tags, ref_tags)
    max_len = max(len(pred_tags), len(ref_tags))
    if max_len == 0:
        return 1.0
    return 1.0 - (dist / max_len)


def compute_teds_batch(
    pred_htmls: Sequence[str],
    ref_htmls: Sequence[str],
) -> float:
    """Average TEDS across a batch of table predictions."""
    if not pred_htmls:
        return 0.0
    total = sum(compute_teds(p, r) for p, r in zip(pred_htmls, ref_htmls))
    return total / len(pred_htmls)
