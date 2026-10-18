import importlib.util
import json
import sys
import tempfile
import types
import numpy as np
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


def _mark_as_kotoba_pipeline(pipe):
    type(pipe).__module__ = "transformers_modules.kotoba_test.synthetic.kotoba_whisper"
    return pipe


def _load_engine_module():
    engine_path = Path(__file__).resolve().parent.parent / "kotoba_transcriber" / "engine.py"
    spec = importlib.util.spec_from_file_location("kotoba_engine_test", engine_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeTensor:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def ne(self, value):
        return f"mask:{value}"


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.prompt_ids = _FakeTensor()

    def get_prompt_ids(self, prompt, return_tensors="pt"):
        return self.prompt_ids


class _FakePunctuator:
    def punctuate(self, text):
        return f"{text}。"


class _FakeModel:
    device = "cuda:0"


class _FakePipeline:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeModel()
        self.punctuator = _FakePunctuator()
        self.feature_extractor = type("_FakeFeatureExtractor", (), {"sampling_rate": 16000})()
        self.calls = []

    def __call__(self, audio_input, **kwargs):
        self.calls.append({"audio_input": audio_input, "kwargs": kwargs})
        return {
            "speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "text/SPEAKER_00": "おはようございます今日は晴れです。",
            "text/SPEAKER_01": "はい。",
            "chunks": [
                {"speaker_id": "SPEAKER_00", "timestamp": [0.0, 1.0], "text": "おはようございます"},
                {"speaker_id": "SPEAKER_00", "timestamp": [1.1, 2.0], "text": "今日は晴れです"},
                {"speaker_id": "SPEAKER_01", "timestamp": [2.5, 3.0], "text": "はい"},
            ],
        }


class TestKotobaEngine(TestCase):
    def test_create_pipeline_uses_custom_kotoba_task(self) -> None:
        module = _load_engine_module()
        captured = {}

        def fake_pipeline_factory(**kwargs):
            captured.update(kwargs)
            return object()

        config = {
            "model_id": module.MODEL_ID,
            "device": "cuda:0",
            "device_pyannote": "cuda:0",
            "torch_dtype": "float16",
            "batch_size": module.DEFAULT_BATCH_SIZE,
            "attn_implementation": "sdpa",
            "model_pyannote": module.DEFAULT_MODEL_PYANNOTE,
            "model_diarizers": module.DEFAULT_MODEL_DIARIZERS,
        }

        with patch.object(module, "_import_pipeline_factory", return_value=fake_pipeline_factory), patch.object(
            module,
            "_get_auth_token",
            return_value=None,
        ):
            module._create_pipeline(config)

        self.assertEqual(captured["task"], "kotoba-whisper")
        self.assertEqual(captured["device_pyannote"], "cuda:0")
        self.assertEqual(captured["model_pyannote"], module.DEFAULT_MODEL_PYANNOTE)

    def test_transcribe_long_audio_writes_speaker_aware_text_and_metadata(self) -> None:
        module = _load_engine_module()
        fake_pipe = _FakePipeline()

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "sample.wav"
            output_path = Path(tmp_dir) / "sample.txt"
            audio_path.write_bytes(b"dummy")

            with patch.object(module, "_get_or_create_pipeline", return_value=fake_pipe), patch.object(
                module,
                "_resolve_pipeline_config",
                return_value={
                    "model_id": module.MODEL_ID,
                    "chunk_length_s": module.DEFAULT_CHUNK_LENGTH_S,
                    "batch_size": module.DEFAULT_BATCH_SIZE,
                    "add_punctuation": True,
                    "num_speakers": None,
                    "min_speakers": None,
                    "max_speakers": None,
                    "stride_length_s": None,
                    "add_silence_start": None,
                    "add_silence_end": None,
                },
            ), patch.object(
                module,
                "_load_audio_input",
                return_value={"array": np.arange(8, dtype=np.float32), "sampling_rate": 16000},
            ):
                result = module.transcribe_long_audio(
                    audio_file=str(audio_path),
                    output_path=str(output_path),
                    prompt="配信者: テスト",
                    header="配信: テスト",
                )

            self.assertEqual(result, str(output_path))
            self.assertTrue(output_path.exists())
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "配信: テスト\n\nSPEAKER_00: おはようございます今日は晴れです。\nSPEAKER_01: はい。",
            )

            metadata_path = output_path.with_name("sample.transcript.json")
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["engine"], "kotoba-whisper")
            self.assertEqual(metadata["model"], "kotoba-tech/kotoba-whisper-v2.2")
            self.assertEqual(len(metadata["segments"]), 2)
            self.assertEqual(metadata["segments"][0]["speaker"], "SPEAKER_00")
            self.assertEqual(metadata["speakers"][0]["text"], "おはようございます今日は晴れです。")

            audio_input = fake_pipe.calls[0]["audio_input"]
            call = fake_pipe.calls[0]["kwargs"]
            self.assertIsInstance(audio_input, dict)
            self.assertEqual(audio_input["sampling_rate"], 16000)
            self.assertIsInstance(audio_input["array"], np.ndarray)
            self.assertEqual(call["chunk_length_s"], module.DEFAULT_CHUNK_LENGTH_S)
            self.assertTrue(call["add_punctuation"])
            self.assertEqual(call["generate_kwargs"]["language"], "ja")
            self.assertEqual(call["generate_kwargs"]["task"], "transcribe")
            self.assertEqual(call["generate_kwargs"]["attention_mask"], "mask:0")

    def test_offload_model_clears_cached_pipeline(self) -> None:
        module = _load_engine_module()
        module._PIPELINE = object()
        module._PIPELINE_SIGNATURE = ("sig",)

        with patch.object(module, "_import_torch", side_effect=RuntimeError("skip torch")):
            module.offload_model()

        self.assertIsNone(module._PIPELINE)
        self.assertIsNone(module._PIPELINE_SIGNATURE)

    def test_patch_pipeline_preprocess_normalizes_final_item_after_zero_sample_tail(self) -> None:
        module = _load_engine_module()
        sample_rate = 16000
        audio_array = np.arange(sample_rate, dtype=np.float32)
        speaker_spans = [(0.0, 1.0), (1.000001, 1.000002)]
        source_payloads = []

        class KotobaWhisperPipeline:
            def preprocess(self, audio_input, **kwargs):
                self.received = (audio_input, kwargs)
                for span_index, (start, end) in enumerate(speaker_spans):
                    start_sample = round(start * sample_rate)
                    end_sample = round(end * sample_rate)
                    segment = audio_input["array"][start_sample:end_sample]
                    if segment.size == 0:
                        continue
                    source_payloads.append(segment)
                    # Source marks the last planned span, not the last emitted
                    # span; a zero-sample final span therefore leaves this false.
                    yield {
                        "input_features": segment,
                        "speaker_span": [start, end],
                        "is_last": span_index == len(speaker_spans) - 1,
                    }

        pipe = _mark_as_kotoba_pipeline(KotobaWhisperPipeline())
        original_input = {"array": audio_array, "sampling_rate": sample_rate}
        self.assertEqual(round(speaker_spans[-1][0] * sample_rate), round(speaker_spans[-1][1] * sample_rate))
        with patch.object(module, "_log") as log:
            module._patch_pipeline_preprocess(pipe)
            result = list(pipe.preprocess(original_input, chunk_length_s=15))

        self.assertEqual(len(result), 1)
        self.assertIs(result[0]["input_features"], source_payloads[0])
        self.assertEqual(result[0]["speaker_span"], [0.0, 1.0])
        self.assertTrue(result[0]["is_last"])
        self.assertEqual(result[0]["input_features"].size, sample_rate)
        self.assertIs(pipe.received[0], original_input)
        self.assertEqual(pipe.received[1], {"chunk_length_s": 15})
        log.assert_called_once_with(
            "Kotoba preprocess protocol normalized: chunks=1 terminal_normalized=true",
            "info",
        )

    def test_patch_pipeline_preprocess_normalizes_last_item_when_final_label_is_empty(self) -> None:
        module = _load_engine_module()

        class KotobaWhisperPipeline:
            def preprocess(self, audio_input):
                # A previous label has an emitted chunk; the final label's
                # timeline is empty and contributes no item.
                yield {"input_features": "earlier-label", "is_last": False}

        pipe = _mark_as_kotoba_pipeline(KotobaWhisperPipeline())
        module._patch_pipeline_preprocess(pipe)
        result = list(pipe.preprocess({"array": np.zeros(1), "sampling_rate": 16000}))

        self.assertEqual([item["input_features"] for item in result], ["earlier-label"])
        self.assertEqual([item["is_last"] for item in result], [True])

    def test_patch_pipeline_preprocess_keeps_partial_batch_and_multispeaker_payloads(self) -> None:
        module = _load_engine_module()
        source_items = [
            {
                "input_features": np.full((1, 4), index, dtype=np.float32),
                "speaker_id": "SPEAKER_00" if index % 2 == 0 else "SPEAKER_01",
                "speaker_span": [float(index), float(index) + 0.5],
                "is_last": False,
            }
            for index in range(9)
        ]

        class KotobaWhisperPipeline:
            def preprocess(self, audio_input):
                self.received_input = audio_input
                yield from source_items

        pipe = _mark_as_kotoba_pipeline(KotobaWhisperPipeline())
        audio_input = {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}
        module._patch_pipeline_preprocess(pipe)
        result = list(pipe.preprocess(audio_input))
        batches = [
            result[offset : offset + module.DEFAULT_BATCH_SIZE]
            for offset in range(0, len(result), module.DEFAULT_BATCH_SIZE)
        ]

        self.assertIs(pipe.received_input, audio_input)
        self.assertEqual([len(batch) for batch in batches], [8, 1])
        self.assertEqual(len(result), len(source_items))
        for index, item in enumerate(result):
            self.assertIs(item["input_features"], source_items[index]["input_features"])
            self.assertEqual(item["speaker_id"], source_items[index]["speaker_id"])
            self.assertEqual(item["speaker_span"], source_items[index]["speaker_span"])
        self.assertEqual([item["is_last"] for item in result], [False] * 8 + [True])

    def test_patch_pipeline_preprocess_leaves_unrelated_pipeline_unchanged(self) -> None:
        module = _load_engine_module()

        class KotobaWhisperPipeline:
            def preprocess(self):
                yield {"input_features": "payload", "is_last": False}

        pipe = KotobaWhisperPipeline()
        original_preprocess = pipe.preprocess.__func__
        module._patch_pipeline_preprocess(pipe)

        self.assertIs(pipe.preprocess.__func__, original_preprocess)
        self.assertFalse(hasattr(pipe, "_codex_terminal_preprocess"))
        self.assertEqual(list(pipe.preprocess()), [{"input_features": "payload", "is_last": False}])

    def test_patch_pipeline_preprocess_preserves_normal_and_existing_end_flags(self) -> None:
        module = _load_engine_module()
        for flags in ((False, True), (True, True)):
            with self.subTest(flags=flags):
                source_items = [
                    {"input_features": "first", "is_last": flags[0]},
                    {"input_features": "second", "is_last": flags[1]},
                ]

                class KotobaWhisperPipeline:
                    def preprocess(self):
                        yield from source_items

                pipe = _mark_as_kotoba_pipeline(KotobaWhisperPipeline())
                module._patch_pipeline_preprocess(pipe)
                with patch.object(module, "_log") as log:
                    result = list(pipe.preprocess())

                self.assertEqual(result, source_items)
                self.assertEqual([item["input_features"] for item in result], ["first", "second"])
                self.assertEqual([item["is_last"] for item in result], list(flags))
                log.assert_not_called()

    def test_patch_pipeline_preprocess_rejects_empty_stream(self) -> None:
        module = _load_engine_module()

        class KotobaWhisperPipeline:
            def preprocess(self):
                if False:
                    yield {}

        pipe = _mark_as_kotoba_pipeline(KotobaWhisperPipeline())
        module._patch_pipeline_preprocess(pipe)

        with self.assertRaisesRegex(module.KotobaPreprocessProtocolError, "no items"):
            list(pipe.preprocess())

    def test_patch_pipeline_preprocess_rejects_non_mapping_items(self) -> None:
        module = _load_engine_module()

        class KotobaWhisperPipeline:
            def preprocess(self):
                yield {"input_features": "valid", "is_last": False}
                yield object()

        pipe = _mark_as_kotoba_pipeline(KotobaWhisperPipeline())
        module._patch_pipeline_preprocess(pipe)
        seen = []

        with self.assertRaisesRegex(module.KotobaPreprocessProtocolError, "non-mapping"):
            for item in pipe.preprocess():
                seen.append(item)

        self.assertEqual([item["input_features"] for item in seen], ["valid"])
        self.assertEqual([item["is_last"] for item in seen], [False])

    def test_patch_pipeline_preprocess_propagates_call_error(self) -> None:
        module = _load_engine_module()

        class KotobaWhisperPipeline:
            def preprocess(self):
                raise TypeError("synthetic preprocess call failure")

        pipe = _mark_as_kotoba_pipeline(KotobaWhisperPipeline())
        module._patch_pipeline_preprocess(pipe)

        with self.assertRaisesRegex(TypeError, "synthetic preprocess call failure"):
            list(pipe.preprocess())

    def test_patch_pipeline_preprocess_propagates_failure_without_terminal_normalization(self) -> None:
        module = _load_engine_module()
        source_items = [
            {"input_features": "first", "is_last": False},
            {"input_features": "second", "is_last": False},
        ]
        failure = ValueError("synthetic preprocess failure")

        class KotobaWhisperPipeline:
            def preprocess(self):
                yield from source_items
                raise failure

        pipe = _mark_as_kotoba_pipeline(KotobaWhisperPipeline())
        module._patch_pipeline_preprocess(pipe)
        seen = []
        with patch.object(module, "_log") as log:
            with self.assertRaises(ValueError) as raised:
                for item in pipe.preprocess():
                    seen.append(item)

        self.assertIs(raised.exception, failure)
        self.assertEqual(seen, [source_items[0]])
        self.assertFalse(any(item.get("is_last") is True for item in seen))
        log.assert_not_called()

    def test_patch_pipeline_preprocess_does_not_release_terminal_pending_item_on_failure(self) -> None:
        module = _load_engine_module()
        failure = ValueError("failure after terminal-looking item")
        terminal_item = {"input_features": "unvalidated-tail", "is_last": True}

        class KotobaWhisperPipeline:
            def preprocess(self):
                yield terminal_item
                raise failure

        pipe = _mark_as_kotoba_pipeline(KotobaWhisperPipeline())
        module._patch_pipeline_preprocess(pipe)

        with self.assertRaises(ValueError) as raised:
            next(iter(pipe.preprocess()))

        self.assertIs(raised.exception, failure)

    def test_patch_pipeline_preprocess_is_idempotent(self) -> None:
        module = _load_engine_module()
        preprocess_calls = []
        source_items = [{"input_features": "only-item", "is_last": False}]

        class SinglePassItems:
            def __init__(self):
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations > 1:
                    raise AssertionError("preprocess output was traversed more than once")
                yield from source_items

        output = SinglePassItems()

        class KotobaWhisperPipeline:
            def preprocess(self):
                preprocess_calls.append(None)
                return output

        pipe = _mark_as_kotoba_pipeline(KotobaWhisperPipeline())
        module._patch_pipeline_preprocess(pipe)
        first_wrapper = pipe.preprocess.__func__
        module._patch_pipeline_preprocess(pipe)

        self.assertIs(pipe.preprocess.__func__, first_wrapper)
        result = list(pipe.preprocess())

        self.assertEqual(preprocess_calls, [None])
        self.assertEqual(output.iterations, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["input_features"], "only-item")
        self.assertTrue(result[0]["is_last"])

    def test_get_or_create_pipeline_installs_preprocess_protocol_patch(self) -> None:
        module = _load_engine_module()

        class KotobaWhisperPipeline:
            def preprocess(self, audio_input, **kwargs):
                self.received = (audio_input, kwargs)
                yield {"input_features": "only-item", "is_last": False}

        pipe = _mark_as_kotoba_pipeline(KotobaWhisperPipeline())
        module._PIPELINE = None
        module._PIPELINE_SIGNATURE = None

        with patch.object(
            module,
            "_resolve_pipeline_config",
            return_value={
                "model_id": module.MODEL_ID,
                "device": "cpu",
                "device_pyannote": "cpu",
                "model_pyannote": module.DEFAULT_MODEL_PYANNOTE,
            },
        ), patch.object(
            module, "_pipeline_signature", return_value=("candidate",)
        ), patch.object(module, "_create_pipeline", return_value=pipe), patch.object(
            module, "_patch_pipeline_forward", side_effect=lambda value: value
        ):
            result = module._get_or_create_pipeline()

        self.assertIs(result, pipe)
        self.assertTrue(list(pipe.preprocess({"array": "payload"}))[0]["is_last"])
        self.assertEqual(pipe.received, ({"array": "payload"}, {}))

    def test_patch_pipeline_postprocess_handles_missing_end_timestamp(self) -> None:
        module = _load_engine_module()

        class _FakeDecodeTokenizer:
            def _decode_asr(self, outputs, **kwargs):
                del outputs, kwargs
                return "", {"chunks": [{"timestamp": [0.25, None], "text": "こんにちは"}]}

        class _FakeConfig:
            max_source_positions = 1500

        class _FakeFeatureExtractor:
            chunk_length = 30

        class _FakePatchedPipeline:
            def __init__(self):
                self.tokenizer = _FakeDecodeTokenizer()
                self.feature_extractor = _FakeFeatureExtractor()
                self.model = type("_FakeModel", (), {"config": _FakeConfig()})()
                self.punctuator = None

            def postprocess(self, model_outputs, **postprocess_parameters):
                raise AssertionError("original postprocess should be replaced")

        pipe = _FakePatchedPipeline()
        module._patch_pipeline_postprocess(pipe)

        result = pipe.postprocess(
            [{"speaker_span": [10.0, 12.0], "speaker_id": "SPEAKER_00"}],
            add_punctuation=False,
            return_language=False,
            return_timestamps=True,
        )

        self.assertEqual(result["speaker_ids"], ["SPEAKER_00"])
        self.assertEqual(result["chunks"][0]["timestamp"], [10.25, 10.25])
        self.assertEqual(result["text/SPEAKER_00"], "こんにちは")

    def test_patch_pyannote_stats_pool_uses_correction_zero_for_single_frame(self) -> None:
        module = _load_engine_module()

        class _FakeTensor:
            def __init__(self):
                self.std_corrections = []

            def size(self, dim):
                self.last_size_dim = dim
                return 1

            def mean(self, dim=-1):
                self.last_mean_dim = dim
                return "mean"

            def std(self, dim=-1, correction=None):
                self.std_corrections.append((dim, correction))
                return f"std:{correction}"

        class _FakeTorch:
            @staticmethod
            def cat(items, dim=-1):
                return {"items": list(items), "dim": dim}

        class _FakeStatsPool:
            def forward(self, sequences, weights=None):
                raise AssertionError("single-frame path should not call original forward")

        fake_pooling = types.ModuleType("pyannote.audio.models.blocks.pooling")
        fake_pooling.StatsPool = _FakeStatsPool

        fake_modules = {
            "pyannote": types.ModuleType("pyannote"),
            "pyannote.audio": types.ModuleType("pyannote.audio"),
            "pyannote.audio.models": types.ModuleType("pyannote.audio.models"),
            "pyannote.audio.models.blocks": types.ModuleType("pyannote.audio.models.blocks"),
            "pyannote.audio.models.blocks.pooling": fake_pooling,
        }

        with patch.dict(sys.modules, fake_modules, clear=False):
            with patch.object(module, "_import_torch", return_value=_FakeTorch()):
                module._patch_pyannote_stats_pool()
                tensor = _FakeTensor()
                result = _FakeStatsPool().forward(tensor)

        self.assertEqual(tensor.std_corrections, [(-1, 0)])
        self.assertEqual(result["items"], ["mean", "std:0"])
        self.assertEqual(result["dim"], -1)

    def test_normalize_speechbrain_redirect_rebinds_pretrained_alias(self) -> None:
        module = _load_engine_module()
        speechbrain_module = types.ModuleType("speechbrain")
        inference_module = types.ModuleType("speechbrain.inference")
        redirect_module = types.ModuleType("speechbrain.pretrained")
        speechbrain_module.pretrained = redirect_module

        def fake_import_module(name):
            if name == "speechbrain":
                return speechbrain_module
            if name == "speechbrain.inference":
                return inference_module
            raise ImportError(name)

        with patch.object(module.importlib, "import_module", side_effect=fake_import_module), patch.dict(
            sys.modules,
            {"speechbrain.pretrained": redirect_module},
            clear=False,
        ):
            module._normalize_speechbrain_redirect()
            self.assertIs(speechbrain_module.pretrained, inference_module)
            self.assertIs(sys.modules["speechbrain.pretrained"], inference_module)

    def test_configure_torch_backends_enables_cudnn_benchmark_for_cuda(self) -> None:
        module = _load_engine_module()
        fake_torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(
                cuda=types.SimpleNamespace(
                    matmul=types.SimpleNamespace(allow_tf32=True),
                ),
                cudnn=types.SimpleNamespace(benchmark=False),
            )
        )

        with patch.object(module, "_import_torch", return_value=fake_torch):
            module._configure_torch_backends({"device": "cuda:0", "device_pyannote": "cuda:0"})

        self.assertFalse(fake_torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(fake_torch.backends.cudnn.allow_tf32)
        self.assertTrue(fake_torch.backends.cudnn.benchmark)

    def test_load_audio_input_returns_writable_array(self) -> None:
        module = _load_engine_module()
        source_audio = np.arange(8, dtype=np.float32)
        source_audio.setflags(write=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "sample.wav"
            audio_path.write_bytes(b"dummy")

            with patch.object(module, "_import_ffmpeg_read", return_value=lambda _data, _rate: source_audio):
                result = module._load_audio_input(audio_path, 16000)

        self.assertEqual(result["sampling_rate"], 16000)
        self.assertTrue(result["array"].flags.writeable)
        self.assertFalse(result["array"] is source_audio)
