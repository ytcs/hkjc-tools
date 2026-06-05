"""Stage 4 — IDENTIFY: bind each tracklet to a runner (saddlecloth number / horse).

Saddlecloth OCR: PARSeq (engine) / EasyOCR (baseline). Silks colour: small CNN or
colour histogram, primed by the race's declared silks. Per-frame OCR is noisy
*votes*; the final mapping is a permutation under one-number-per-runner (Hungarian).
Heavy imports lazy.
"""
from __future__ import annotations

from typing import Optional

from .schema import IdentityVote, Tracklet


class SaddleclothOCR:
    """OCR on the saddlecloth ROI only (crop top of the box). Lazy heavy import."""

    def __init__(self, engine: str = "parseq") -> None:
        self.engine = engine     # "parseq" (SOTA) | "easyocr" (baseline)
        self._model = None

    def read(self, roi) -> tuple[Optional[int], float]:
        """Return (number, confidence) for one saddlecloth crop. STUB."""
        raise NotImplementedError(
            "Stub. Lazy-import PARSeq/EasyOCR; OCR the saddlecloth crop; parse to int."
        )


class SilksClassifier:
    """Classify jacket silks, primed by the race's declared per-runner silks. STUB."""

    def __init__(self) -> None:
        self._model = None

    def classify(self, roi) -> tuple[Optional[str], float]:
        raise NotImplementedError("Stub — silks colour classifier (CNN/histogram).")


def vote_tracklet(
    tracklet: Tracklet,
    ocr: SaddleclothOCR,
    silks: Optional[SilksClassifier] = None,
) -> IdentityVote:
    """Accumulate OCR + silks votes over a tracklet's whole life. STUB."""
    raise NotImplementedError(
        "Stub — run OCR/silks on each detection's ROI, aggregate to one robust vote."
    )


def assign_runners(
    votes: list[IdentityVote],
    field_size: int,
    declared_numbers: Optional[list[int]] = None,
) -> dict[int, int]:
    """Global one-number-per-runner assignment: track_id -> saddlecloth number.

    Use Hungarian on a (track_id x candidate-number) cost from fused OCR/silks
    confidence, constrained to a permutation over the declared field. STUB.
    """
    raise NotImplementedError(
        "Stub — build cost matrix from votes; scipy.optimize.linear_sum_assignment\n"
        "under the one-number-per-runner / known-field-size constraint."
    )
