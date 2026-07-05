from __future__ import annotations

import copy
import json
from pathlib import Path

from kotoba_transcriber.hallucination_snapshot import annotate_transcript, annotate_transcript_file


def _snapshot() -> dict:
    return {
        "schema_version": 1,
        "snapshot_id": "snapshot-test",
        "top_candidates": [
            {
                "file_path": "chan/sample.transcript.json",
                "segment_index": 0,
                "start": 0.0,
                "text": "ありがとうございました",
                "score": 0.95,
                "bucket": "high_confidence_hallucination_candidate",
                "reasons": ["dominant phrase match: ありがとうございました"],
                "evidence_families": [
                    "phrase_dominance",
                    "boundary_or_short",
                    "speaker_rarity_or_enrichment",
                ],
                "summary_omit_candidate": True,
                "snapshot_id": "snapshot-test",
            }
        ],
    }


def _transcript() -> dict:
    return {
        "segments": [
            {"start": 0.0, "end": 1.0, "speaker": "S0", "text": "ありがとうございました"},
            {"start": 2.0, "end": 3.0, "speaker": "S1", "text": "残す本文"},
        ]
    }


def test_annotator_adds_hallucination_object_to_matching_segments():
    annotated = annotate_transcript(_transcript(), _snapshot(), transcript_path=Path("sample.transcript.json"))

    assert annotated["segments"][0]["hallucination"]["score"] == 0.95
    assert annotated["segments"][0]["hallucination"]["summary_omit_candidate"] is True
    assert annotated["segments"][0]["hallucination"]["match_quality"] == "file_path_segment_index"


def test_input_transcript_is_not_modified():
    transcript = _transcript()
    before = copy.deepcopy(transcript)

    annotate_transcript(transcript, _snapshot(), transcript_path=Path("sample.transcript.json"))

    assert transcript == before


def test_unmatched_segments_remain_keep_safely():
    annotated = annotate_transcript(_transcript(), _snapshot(), transcript_path=Path("different.json"))

    assert annotated["segments"][1]["hallucination"]["bucket"] == "keep"
    assert annotated["segments"][1]["hallucination"]["score"] == 0.0
    assert annotated["segments"][1]["hallucination"]["summary_omit_candidate"] is False


def test_summary_omit_candidate_propagates_from_snapshot(tmp_path: Path):
    transcript_path = tmp_path / "sample.transcript.json"
    snapshot_path = tmp_path / "snapshot.json"
    output_path = tmp_path / "annotated.json"
    transcript_path.write_text(json.dumps(_transcript(), ensure_ascii=False), encoding="utf-8")
    snapshot_path.write_text(json.dumps(_snapshot(), ensure_ascii=False), encoding="utf-8")

    annotate_transcript_file(transcript_path, snapshot_path, output_path)

    original = json.loads(transcript_path.read_text(encoding="utf-8"))
    annotated = json.loads(output_path.read_text(encoding="utf-8"))
    assert "hallucination" not in original["segments"][0]
    assert annotated["segments"][0]["hallucination"]["summary_omit_candidate"] is True
