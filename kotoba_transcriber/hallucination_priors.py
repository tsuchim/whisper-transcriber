#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Annotation-only hallucination review priors for transcript metadata."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ENV_PRIORS_PATH = "KOTOBA_HALLUCINATION_PRIORS_PATH"
CONSUMER_CONTRACT = "metadata-segment-priors-v1"
POLICY = "annotation_only"
STRONG_PHRASES = {"ごめん", "ありがとうございました"}


class HallucinationPriorsError(ValueError):
    """Raised when a priors snapshot cannot be consumed safely."""


@dataclass(frozen=True)
class PhraseDefinition:
    text: str
    normalized_text: str


@dataclass(frozen=True)
class HallucinationPriors:
    schema_version: int
    generated_at: str | None
    minimum_consumer_contract: str
    short_segment_seconds: float
    phrase_definitions: tuple[PhraseDefinition, ...]
    phrase_priors: dict[str, dict[str, Any]]


def load_priors(path: str | os.PathLike[str]) -> HallucinationPriors:
    """Load and validate a priors snapshot."""

    snapshot_path = Path(path)
    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HallucinationPriorsError(f"Invalid hallucination priors JSON: {snapshot_path}") from exc

    if not isinstance(data, dict):
        raise HallucinationPriorsError("Hallucination priors snapshot must be a JSON object")
    if data.get("schema_version") != 1:
        raise HallucinationPriorsError("Unsupported hallucination priors schema_version")

    compatibility = data.get("compatibility")
    if not isinstance(compatibility, dict):
        raise HallucinationPriorsError("Missing hallucination priors compatibility block")
    contract = compatibility.get("minimum_consumer_contract")
    if contract != CONSUMER_CONTRACT:
        raise HallucinationPriorsError("Unsupported hallucination priors consumer contract")
    if compatibility.get("does_not_define_deletion_rules") is not True:
        raise HallucinationPriorsError("Hallucination priors must not define deletion rules")

    definitions: list[PhraseDefinition] = []
    for item in data.get("phrase_definitions") or []:
        if not isinstance(item, dict):
            continue
        if item.get("is_deletion_rule") is True:
            raise HallucinationPriorsError("Phrase definitions must not contain deletion rules")
        text = str(item.get("text") or "")
        normalized_text = str(item.get("normalized_text") or text)
        if text:
            definitions.append(PhraseDefinition(text=text, normalized_text=normalized_text))

    thresholds = data.get("duration_thresholds") if isinstance(data.get("duration_thresholds"), dict) else {}
    short_segment_seconds = _as_float(thresholds.get("short_segment_seconds"), 2.0)
    phrase_priors = data.get("phrase_priors") if isinstance(data.get("phrase_priors"), dict) else {}

    return HallucinationPriors(
        schema_version=1,
        generated_at=str(data["generated_at"]) if data.get("generated_at") else None,
        minimum_consumer_contract=contract,
        short_segment_seconds=short_segment_seconds,
        phrase_definitions=tuple(definitions),
        phrase_priors=phrase_priors,
    )


def annotate_segments_from_env(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate segments only when KOTOBA_HALLUCINATION_PRIORS_PATH is set."""

    priors_path = os.getenv(ENV_PRIORS_PATH)
    if not priors_path:
        return segments
    return annotate_segments(segments, load_priors(priors_path))


def annotate_segments(
    segments: list[dict[str, Any]],
    priors: HallucinationPriors,
) -> list[dict[str, Any]]:
    """Return metadata segments with annotation-only hallucination risk fields."""

    speaker_counts = _speaker_counts(segments)
    annotated: list[dict[str, Any]] = []
    total_segments = max(len(segments), 1)

    for index, segment in enumerate(segments):
        item = dict(segment)
        risk = _segment_risk(item, index, total_segments, speaker_counts, priors)
        if risk is not None:
            item["hallucination_risk"] = risk
        annotated.append(item)
    return annotated


def _segment_risk(
    segment: dict[str, Any],
    index: int,
    total_segments: int,
    speaker_counts: dict[str, int],
    priors: HallucinationPriors,
) -> dict[str, Any] | None:
    text = str(segment.get("text") or "")
    if not text:
        return None

    duration = max(0.0, _as_float(segment.get("end"), 0.0) - _as_float(segment.get("start"), 0.0))
    short_segment = duration <= priors.short_segment_seconds
    rare_speaker = speaker_counts.get(str(segment.get("speaker") or ""), 0) <= 1 and total_segments >= 3
    boundary_segment = index == 0 or index == total_segments - 1
    span_risks = _span_risks(text, duration, priors)

    reasons: list[str] = []
    score = 0.0
    if span_risks:
        score = max(_as_float(span["score"], 0.0) for span in span_risks)
        reasons.extend(_unique(reason for span in span_risks for reason in span["reasons"]))

    normalized_len = len(_normalize_text(text))
    if short_segment and normalized_len <= 8:
        score = max(score, 0.58)
        reasons.append("short_segment")
        if not span_risks:
            reasons.append("phrase_free_short_segment")

    if rare_speaker and score:
        score = min(0.99, score + 0.06)
        reasons.append("rare_speaker")
    if boundary_segment and score >= 0.55:
        score = min(0.99, score + 0.03)
        reasons.append("boundary_segment")

    if score < 0.45:
        return None

    return {
        "candidate": True,
        "score": round(score, 2),
        "bucket": _bucket(score),
        "policy": POLICY,
        "reasons": _unique(reasons),
        "span_risks": span_risks,
        "model": {
            "priors_schema_version": priors.schema_version,
            "minimum_consumer_contract": priors.minimum_consumer_contract,
            "snapshot_generated_at": priors.generated_at,
            "score_kind": "conservative_review_prior",
        },
    }


def _span_risks(text: str, duration: float, priors: HallucinationPriors) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    normalized_segment = _normalize_text(text)

    for phrase in priors.phrase_definitions:
        if phrase.normalized_text not in STRONG_PHRASES:
            continue
        for start, end in _find_occurrences(text, phrase.text):
            reasons = ["phrase_match"]
            share = len(_normalize_text(text[start:end])) / max(len(normalized_segment), 1)
            phrase_prior = priors.phrase_priors.get(phrase.normalized_text, {})
            score = 0.62
            score += 0.16 * _as_float(phrase_prior.get("dominance_rate_among_matches"), 0.0)
            score += 0.10 * _as_float(phrase_prior.get("short_segment_rate_among_matches"), 0.0)
            if share >= 0.7:
                score += 0.12
                reasons.append("phrase_dominance")
            if duration <= priors.short_segment_seconds:
                score += 0.05
                reasons.append("short_segment")
            score = min(score, 0.99)
            risks.append(
                {
                    "candidate": True,
                    "score": round(score, 2),
                    "bucket": _bucket(score),
                    "text": text[start:end],
                    "normalized_text": phrase.normalized_text,
                    "char_start": start,
                    "char_end": end,
                    "reasons": reasons,
                    "policy": POLICY,
                }
            )

    risks.sort(key=lambda risk: (risk["char_start"], risk["char_end"], risk["normalized_text"]))
    return risks


def _find_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    occurrences: list[tuple[int, int]] = []
    start = 0
    while True:
        found = text.find(needle, start)
        if found < 0:
            break
        end = found + len(needle)
        occurrences.append((found, end))
        start = end
    return occurrences


def _speaker_counts(segments: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for segment in segments:
        speaker = str(segment.get("speaker") or "")
        counts[speaker] = counts.get(speaker, 0) + 1
    return counts


def _bucket(score: float) -> str:
    if score >= 0.85:
        return "high_candidate"
    if score >= 0.65:
        return "review_candidate"
    if score >= 0.45:
        return "low_candidate"
    return "control"


def _normalize_text(text: str) -> str:
    return "".join(str(text).split())


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unique(items: Any) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for item in items:
        text = str(item)
        if text and text not in seen:
            values.append(text)
            seen.add(text)
    return values
