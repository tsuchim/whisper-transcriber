#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Annotate transcript JSON from a live-scraping hallucination snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    return "".join(str(text or "").split())


def load_snapshot(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("hallucination snapshot must be a JSON object")
    if data.get("schema_version") != 1:
        raise ValueError("unsupported hallucination snapshot schema_version")
    if not isinstance(data.get("top_candidates"), list):
        raise ValueError("hallucination snapshot is missing top_candidates")
    return data


def _default_annotation(snapshot_id: str | None) -> dict[str, Any]:
    return {
        "score": 0.0,
        "bucket": "keep",
        "reasons": [],
        "evidence_families": [],
        "summary_omit_candidate": False,
        "snapshot_id": snapshot_id,
        "match_quality": "unmatched_keep_default",
    }


def _candidate_annotation(candidate: dict[str, Any], match_quality: str) -> dict[str, Any]:
    return {
        "score": float(candidate.get("score") or 0.0),
        "bucket": str(candidate.get("bucket") or "keep"),
        "reasons": list(candidate.get("reasons") or []),
        "evidence_families": list(candidate.get("evidence_families") or []),
        "summary_omit_candidate": bool(candidate.get("summary_omit_candidate") is True),
        "snapshot_id": candidate.get("snapshot_id"),
        "match_quality": match_quality,
    }


def _candidate_matches_path(candidate: dict[str, Any], transcript_path: Path) -> bool:
    file_path = str(candidate.get("file_path") or candidate.get("relative_file_id") or "")
    if not file_path:
        return False
    return file_path == str(transcript_path) or file_path.endswith(transcript_path.name)


def _find_candidate(
    segment: dict[str, Any],
    index: int,
    transcript_path: Path,
    candidates: list[dict[str, Any]],
    *,
    timestamp_tolerance: float,
) -> tuple[dict[str, Any] | None, str]:
    for candidate in candidates:
        if not _candidate_matches_path(candidate, transcript_path):
            continue
        if int(candidate.get("segment_index", -1)) == index:
            return candidate, "file_path_segment_index"

    normalized = normalize_text(str(segment.get("text") or ""))
    start = _as_float(segment.get("start"), 0.0)
    for candidate in candidates:
        if normalize_text(str(candidate.get("text") or "")) != normalized:
            continue
        candidate_start = _as_float(candidate.get("start"), 0.0)
        if abs(candidate_start - start) <= timestamp_tolerance:
            return candidate, "normalized_text_timestamp"

    return None, "unmatched_keep_default"


def annotate_transcript(
    transcript: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    transcript_path: Path,
    timestamp_tolerance: float = 0.75,
) -> dict[str, Any]:
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        segments = []

    candidates = [item for item in snapshot.get("top_candidates", []) if isinstance(item, dict)]
    snapshot_id = snapshot.get("snapshot_id")
    annotated_segments: list[dict[str, Any]] = []
    for index, raw in enumerate(segments):
        if not isinstance(raw, dict):
            annotated_segments.append(raw)
            continue
        segment = dict(raw)
        candidate, match_quality = _find_candidate(
            segment,
            index,
            transcript_path,
            candidates,
            timestamp_tolerance=timestamp_tolerance,
        )
        if candidate is None:
            segment["hallucination"] = _default_annotation(str(snapshot_id) if snapshot_id else None)
        else:
            segment["hallucination"] = _candidate_annotation(candidate, match_quality)
        annotated_segments.append(segment)

    annotated = dict(transcript)
    annotated["segments"] = annotated_segments
    annotated["hallucination_annotation"] = {
        "snapshot_id": snapshot_id,
        "policy": "annotation_only",
        "source_transcript_preserved": True,
    }
    return annotated


def annotate_transcript_file(transcript_path: Path, snapshot_path: Path, output_path: Path) -> dict[str, Any]:
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(transcript, dict):
        raise ValueError("transcript JSON must be an object")
    annotated = annotate_transcript(transcript, load_snapshot(snapshot_path), transcript_path=transcript_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(annotated, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return annotated


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotated = annotate_transcript_file(args.transcript, args.snapshot, args.output)
    print(f"wrote {args.output}")
    print(f"segments={len(annotated.get('segments') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
