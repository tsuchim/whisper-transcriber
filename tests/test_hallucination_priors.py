import importlib
import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from kotoba_transcriber.hallucination_priors import (
    CONSUMER_CONTRACT,
    HallucinationPriorsError,
    annotate_segments,
    load_priors,
)


def _snapshot(**overrides):
    data = {
        "schema_version": 1,
        "generated_at": "2026-07-04T08:04:57Z",
        "compatibility": {
            "minimum_consumer_contract": CONSUMER_CONTRACT,
            "does_not_define_deletion_rules": True,
        },
        "duration_thresholds": {
            "short_segment_seconds": 2.0,
        },
        "phrase_definitions": [
            {
                "id": "phrase_1",
                "is_deletion_rule": False,
                "text": "ごめん",
                "normalized_text": "ごめん",
            },
            {
                "id": "phrase_2",
                "is_deletion_rule": False,
                "text": "ありがとうございました",
                "normalized_text": "ありがとうございました",
            },
            {
                "id": "phrase_3",
                "is_deletion_rule": False,
                "text": "チャンネル登録",
                "normalized_text": "チャンネル登録",
            },
            {
                "id": "phrase_4",
                "is_deletion_rule": False,
                "text": "高評価",
                "normalized_text": "高評価",
            },
        ],
        "phrase_priors": {
            "ごめん": {
                "dominance_rate_among_matches": 0.8,
                "short_segment_rate_among_matches": 0.73,
            },
            "ありがとうございました": {
                "dominance_rate_among_matches": 0.71,
                "short_segment_rate_among_matches": 0.6,
            },
            "チャンネル登録": {
                "dominance_rate_among_matches": 0.0,
                "short_segment_rate_among_matches": 0.06,
            },
            "高評価": {
                "dominance_rate_among_matches": 0.17,
                "short_segment_rate_among_matches": 0.33,
            },
        },
    }
    data.update(overrides)
    return data


def _write_snapshot(directory, data=None):
    path = Path(directory) / "priors.json"
    path.write_text(json.dumps(data or _snapshot(), ensure_ascii=False), encoding="utf-8")
    return path


class _FakeFeatureExtractor:
    sampling_rate = 16000


class _FakePipeline:
    feature_extractor = _FakeFeatureExtractor()

    def __call__(self, audio_input, **kwargs):
        del audio_input, kwargs
        return {
            "speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "text/SPEAKER_00": "今日はありがとうございました。またね",
            "text/SPEAKER_01": "はい。",
            "chunks": [
                {
                    "speaker_id": "SPEAKER_00",
                    "timestamp": [0.0, 1.0],
                    "text": "今日はありがとうございました。またね",
                },
                {
                    "speaker_id": "SPEAKER_01",
                    "timestamp": [3.0, 3.5],
                    "text": "はい。",
                },
            ],
        }


def _engine_config(module):
    return {
        "model_id": module.MODEL_ID,
        "chunk_length_s": module.DEFAULT_CHUNK_LENGTH_S,
        "batch_size": module.DEFAULT_BATCH_SIZE,
        "add_punctuation": False,
        "num_speakers": None,
        "min_speakers": None,
        "max_speakers": None,
        "stride_length_s": None,
        "add_silence_start": None,
        "add_silence_end": None,
    }


class TestHallucinationPriors(TestCase):
    def test_load_priors_validates_snapshot_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            priors = load_priors(_write_snapshot(tmp_dir))

        self.assertEqual(priors.schema_version, 1)
        self.assertEqual(priors.minimum_consumer_contract, CONSUMER_CONTRACT)
        self.assertEqual(priors.short_segment_seconds, 2.0)

    def test_load_priors_rejects_incompatible_snapshot(self):
        bad_contract = _snapshot(
            compatibility={
                "minimum_consumer_contract": "different-contract",
                "does_not_define_deletion_rules": True,
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(HallucinationPriorsError):
                load_priors(_write_snapshot(tmp_dir, bad_contract))

        deletion_rule = _snapshot(
            compatibility={
                "minimum_consumer_contract": CONSUMER_CONTRACT,
                "does_not_define_deletion_rules": False,
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(HallucinationPriorsError):
                load_priors(_write_snapshot(tmp_dir, deletion_rule))

    def test_span_risks_include_phrase_offsets_without_changing_text(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            priors = load_priors(_write_snapshot(tmp_dir))

        text = "今日はありがとうございました。またね"
        segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": text}]
        annotated = annotate_segments(segments, priors)

        self.assertEqual(annotated[0]["text"], text)
        span = annotated[0]["hallucination_risk"]["span_risks"][0]
        self.assertEqual(span["text"], "ありがとうございました")
        self.assertEqual(span["char_start"], text.index("ありがとうございました"))
        self.assertEqual(span["char_end"], text.index("ありがとうございました") + len("ありがとうございました"))
        self.assertEqual(span["policy"], "annotation_only")

    def test_phrase_free_short_segment_can_be_candidate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            priors = load_priors(_write_snapshot(tmp_dir))

        annotated = annotate_segments(
            [
                {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "今日は晴れです。"},
                {"start": 4.0, "end": 4.5, "speaker": "SPEAKER_01", "text": "はい。"},
                {"start": 5.0, "end": 8.0, "speaker": "SPEAKER_00", "text": "次に進みます。"},
            ],
            priors,
        )

        risk = annotated[1]["hallucination_risk"]
        self.assertTrue(risk["candidate"])
        self.assertIn("phrase_free_short_segment", risk["reasons"])
        self.assertEqual(risk["span_risks"], [])

    def test_channel_registration_and_high_rating_are_not_direct_removal_cues(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            priors = load_priors(_write_snapshot(tmp_dir))

        text = "チャンネル登録と高評価お願いします"
        annotated = annotate_segments(
            [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": text}],
            priors,
        )

        self.assertEqual(annotated[0]["text"], text)
        self.assertNotIn("hallucination_risk", annotated[0])

    def test_engine_behavior_is_unchanged_when_env_var_is_unset(self):
        module = importlib.import_module("kotoba_transcriber.engine")
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "sample.wav"
            output_path = Path(tmp_dir) / "sample.txt"
            audio_path.write_bytes(b"dummy")

            with patch.dict("os.environ", {"KOTOBA_HALLUCINATION_PRIORS_PATH": ""}), patch.object(
                module, "_get_or_create_pipeline", return_value=_FakePipeline()
            ), patch.object(module, "_resolve_pipeline_config", return_value=_engine_config(module)), patch.object(
                module, "_load_audio_input", return_value={"array": [], "sampling_rate": 16000}
            ):
                module.transcribe_long_audio(str(audio_path), output_path=str(output_path))

            rendered_text = output_path.read_text(encoding="utf-8")
            metadata = json.loads(output_path.with_name("sample.transcript.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["text"], rendered_text)
        self.assertNotIn("hallucination_risk", metadata["segments"][0])
        self.assertNotIn("hallucination_risk", metadata["segments"][1])

    def test_engine_annotation_preserves_segment_and_rendered_text(self):
        module = importlib.import_module("kotoba_transcriber.engine")
        with tempfile.TemporaryDirectory() as tmp_dir:
            priors_path = _write_snapshot(tmp_dir)
            audio_path = Path(tmp_dir) / "sample.wav"
            output_path = Path(tmp_dir) / "sample.txt"
            audio_path.write_bytes(b"dummy")

            with patch.dict("os.environ", {"KOTOBA_HALLUCINATION_PRIORS_PATH": str(priors_path)}), patch.object(
                module, "_get_or_create_pipeline", return_value=_FakePipeline()
            ), patch.object(module, "_resolve_pipeline_config", return_value=_engine_config(module)), patch.object(
                module, "_load_audio_input", return_value={"array": [], "sampling_rate": 16000}
            ):
                module.transcribe_long_audio(str(audio_path), output_path=str(output_path))

            rendered_text = output_path.read_text(encoding="utf-8")
            metadata = json.loads(output_path.with_name("sample.transcript.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["text"], rendered_text)
        self.assertEqual(metadata["segments"][0]["text"], "今日はありがとうございました。またね")
        self.assertEqual(metadata["segments"][1]["text"], "はい。")
        self.assertIn("hallucination_risk", metadata["segments"][0])
        self.assertIn("hallucination_risk", metadata["segments"][1])

    def test_unit_tests_do_not_import_heavy_model_packages(self):
        heavy_modules = {
            "diarizers",
            "pyannote",
            "torch",
            "transformers",
            "whisper",
        }
        self.assertTrue(heavy_modules.isdisjoint(set(sys.modules)))
