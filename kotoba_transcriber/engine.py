#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kotoba-whisper v2.2 pipeline wrapper with diarization-aware output."""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import os
import sys
from pathlib import Path
from threading import Lock
from types import MethodType
from typing import Any, Optional


MODEL_ID = "kotoba-tech/kotoba-whisper-v2.2"
ENGINE_ID = "kotoba-whisper"
DEFAULT_CHUNK_LENGTH_S = 15
DEFAULT_BATCH_SIZE = 8
DEFAULT_MODEL_PYANNOTE = "pyannote/speaker-diarization-3.1"
DEFAULT_MODEL_DIARIZERS = "diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn"
DEFAULT_MERGE_GAP_S = 0.75

_logger: Optional[logging.Logger] = None
_logger_lock = Lock()
_pipeline_lock = Lock()
_PIPELINE = None
_PIPELINE_SIGNATURE: Optional[tuple[Any, ...]] = None


def _configure_stdio_for_windows() -> None:
    if sys.platform != "win32":
        return
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


def set_logger(logger: logging.Logger) -> None:
    global _logger
    with _logger_lock:
        _logger = logger


def _log(message: str, level: str = "info") -> None:
    with _logger_lock:
        logger = _logger

    if logger is not None:
        method = getattr(logger, level, None)
        if callable(method):
            method(message)
        else:
            logger.info(message)
        return

    if level == "debug":
        return
    print(message)


def _import_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch が必要です。`pip install -r requirements.txt` を実行してください。") from exc
    return torch


def _import_pipeline_factory():
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError("transformers が必要です。`pip install -r requirements.txt` を実行してください。") from exc
    return hf_pipeline


def _flash_attention_2_available() -> bool:
    try:
        from transformers.utils import is_flash_attn_2_available  # type: ignore
    except Exception:
        return False
    try:
        return bool(is_flash_attn_2_available())
    except Exception:
        return False


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _resolve_device_name(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    torch = _import_torch()
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_torch_dtype(explicit: Optional[str], device_name: str):
    torch = _import_torch()
    if explicit:
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        key = str(explicit).strip().lower()
        if key not in mapping:
            raise RuntimeError(f"未対応の torch dtype です: {explicit}")
        return mapping[key], key
    if device_name.startswith("cuda"):
        return torch.float16, "float16"
    return torch.float32, "float32"


def _resolve_attention_implementation(device_name: str, explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    if not device_name.startswith("cuda"):
        return "eager"
    if _flash_attention_2_available():
        return "flash_attention_2"
    return "sdpa"


def _resolve_pipeline_config(**overrides: Any) -> dict[str, Any]:
    device_name = _resolve_device_name(_env("KOTOBA_DEVICE"))
    torch_dtype, torch_dtype_name = _resolve_torch_dtype(_env("KOTOBA_TORCH_DTYPE"), device_name)

    config = {
        "model_id": str(overrides.get("model_id") or _env("KOTOBA_MODEL_ID", MODEL_ID)),
        "device": str(overrides.get("device") or _env("KOTOBA_DEVICE", device_name)),
        "device_pyannote": str(
            overrides.get("device_pyannote") or _env("KOTOBA_DEVICE_PYANNOTE", device_name)
        ),
        "torch_dtype": overrides.get("torch_dtype") or torch_dtype,
        "torch_dtype_name": str(overrides.get("torch_dtype_name") or torch_dtype_name),
        "attn_implementation": overrides.get("attn_implementation")
        or _resolve_attention_implementation(device_name, _env("KOTOBA_ATTN_IMPLEMENTATION")),
        "batch_size": int(overrides.get("batch_size") or _env("KOTOBA_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
        "chunk_length_s": int(
            overrides.get("chunk_length_s") or _env("KOTOBA_CHUNK_LENGTH_S", str(DEFAULT_CHUNK_LENGTH_S))
        ),
        "stride_length_s": _as_optional_float(
            overrides.get("stride_length_s") if "stride_length_s" in overrides else _env("KOTOBA_STRIDE_LENGTH_S")
        ),
        "add_punctuation": _as_bool(
            overrides.get("add_punctuation")
            if "add_punctuation" in overrides
            else _env("KOTOBA_ADD_PUNCTUATION", "true"),
            True,
        ),
        "num_speakers": _as_optional_int(
            overrides.get("num_speakers") if "num_speakers" in overrides else _env("KOTOBA_NUM_SPEAKERS")
        ),
        "min_speakers": _as_optional_int(
            overrides.get("min_speakers") if "min_speakers" in overrides else _env("KOTOBA_MIN_SPEAKERS")
        ),
        "max_speakers": _as_optional_int(
            overrides.get("max_speakers") if "max_speakers" in overrides else _env("KOTOBA_MAX_SPEAKERS")
        ),
        "add_silence_start": _as_optional_float(
            overrides.get("add_silence_start")
            if "add_silence_start" in overrides
            else _env("KOTOBA_ADD_SILENCE_START")
        ),
        "add_silence_end": _as_optional_float(
            overrides.get("add_silence_end")
            if "add_silence_end" in overrides
            else _env("KOTOBA_ADD_SILENCE_END")
        ),
        "model_pyannote": str(
            overrides.get("model_pyannote") or _env("KOTOBA_MODEL_PYANNOTE", DEFAULT_MODEL_PYANNOTE)
        ),
        "model_diarizers": str(
            overrides.get("model_diarizers") or _env("KOTOBA_MODEL_DIARIZERS", DEFAULT_MODEL_DIARIZERS)
        ),
    }
    return config


def _pipeline_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        config["model_id"],
        config["device"],
        config["device_pyannote"],
        config["torch_dtype_name"],
        config["attn_implementation"],
        config["batch_size"],
        config["model_pyannote"],
        config["model_diarizers"],
    )


def _get_auth_token() -> Optional[str]:
    return (
        _env("HUGGINGFACE_HUB_TOKEN")
        or _env("HF_TOKEN")
        or _env("HUGGINGFACE_TOKEN")
    )


def _create_pipeline(config: dict[str, Any]):
    hf_pipeline = _import_pipeline_factory()
    model_kwargs = {
        "low_cpu_mem_usage": True,
    }
    if config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = config["attn_implementation"]

    token = _get_auth_token()
    pipeline_kwargs = {
        # kotoba-whisper-v2.2 は Hub 上で custom pipeline 名を公開している。
        # 標準 ASR task を指定すると remote pipeline が選ばれず、
        # diarization / punctuation 用の拡張引数が無視される。
        "task": "kotoba-whisper",
        "model": config["model_id"],
        "tokenizer": config["model_id"],
        "feature_extractor": config["model_id"],
        "trust_remote_code": True,
        "device": config["device"],
        "device_pyannote": config["device_pyannote"],
        "torch_dtype": config["torch_dtype"],
        "batch_size": config["batch_size"],
        "model_pyannote": config["model_pyannote"],
        "model_diarizers": config["model_diarizers"],
        "model_kwargs": model_kwargs,
    }
    if token:
        pipeline_kwargs["token"] = token

    try:
        return hf_pipeline(**pipeline_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "kotoba-whisper v2.2 pipeline の初期化に失敗しました。"
            " `pip install -r requirements.txt` と `hf auth login`、"
            " ならびに pyannote モデル利用規約の承諾状態を確認してください。"
        ) from exc


def _get_or_create_pipeline(**overrides: Any):
    global _PIPELINE, _PIPELINE_SIGNATURE

    config = _resolve_pipeline_config(**overrides)
    signature = _pipeline_signature(config)

    with _pipeline_lock:
        if _PIPELINE is not None and _PIPELINE_SIGNATURE == signature:
            return _PIPELINE

        _log(
            "kotoba-whisper v2.2 pipeline をロード中: "
            f"model={config['model_id']} device={config['device']} diarization={config['model_pyannote']}",
        )
        _PIPELINE = _create_pipeline(config)
        _patch_pipeline_postprocess(_PIPELINE)
        _PIPELINE_SIGNATURE = signature
        return _PIPELINE


def offload_model() -> None:
    global _PIPELINE, _PIPELINE_SIGNATURE
    with _pipeline_lock:
        _PIPELINE = None
        _PIPELINE_SIGNATURE = None

    try:
        torch = _import_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    _log("kotoba-whisper pipeline を解放しました", "debug")


def restore_model() -> None:
    _get_or_create_pipeline()


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.transcript.json")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _patch_pipeline_postprocess(pipe: Any) -> Any:
    if getattr(pipe, "_codex_safe_postprocess", False):
        return pipe

    original_postprocess = getattr(pipe, "postprocess", None)
    original_func = getattr(original_postprocess, "__func__", original_postprocess)
    remote_globals = getattr(original_func, "__globals__", {})
    punctuator_cls = remote_globals.get("Punctuator")

    def safe_postprocess(self: Any, model_outputs: list[dict[str, Any]], **postprocess_parameters: Any) -> dict[str, Any]:
        if postprocess_parameters.get("add_punctuation") and getattr(self, "punctuator", None) is None and punctuator_cls:
            self.punctuator = punctuator_cls()

        outputs: dict[str, Any] = {"chunks": []}
        time_precision = self.feature_extractor.chunk_length / self.model.config.max_source_positions

        for output in model_outputs:
            _text, chunks = self.tokenizer._decode_asr(
                [output],
                return_language=postprocess_parameters.get("return_language", False),
                return_timestamps=postprocess_parameters.get("return_timestamps", True),
                time_precision=time_precision,
            )
            speaker_span = output.get("speaker_span") or [0.0, 0.0]
            span_start = _safe_float(speaker_span[0] if len(speaker_span) > 0 else 0.0)
            chunk_items = chunks.get("chunks") if isinstance(chunks, dict) else []
            if not isinstance(chunk_items, list):
                continue

            for chunk in chunk_items:
                if not isinstance(chunk, dict):
                    continue
                timestamp = chunk.get("timestamp") or [0.0, 0.0]
                rel_start = _safe_float(timestamp[0] if len(timestamp) > 0 else 0.0)
                rel_end = _safe_float(timestamp[1] if len(timestamp) > 1 else rel_start, rel_start)
                if rel_end < rel_start:
                    rel_end = rel_start

                item = dict(chunk)
                item["timestamp"] = [
                    round(span_start + rel_start, 2),
                    round(span_start + rel_end, 2),
                ]
                item["speaker_id"] = output.get("speaker_id")
                outputs["chunks"].append(item)

        outputs["speaker_ids"] = sorted(
            {item["speaker_id"] for item in outputs["chunks"] if item.get("speaker_id") is not None}
        )
        for speaker_id in outputs["speaker_ids"]:
            speaker_chunks = sorted(
                [item for item in outputs["chunks"] if item.get("speaker_id") == speaker_id],
                key=lambda item: _safe_float((item.get("timestamp") or [0.0])[0]),
            )
            outputs[f"chunks/{speaker_id}"] = speaker_chunks
            speaker_text = "".join(str(item.get("text") or "") for item in speaker_chunks)
            if postprocess_parameters.get("add_punctuation") and getattr(self, "punctuator", None) is not None:
                speaker_text = self.punctuator.punctuate(speaker_text)
            outputs[f"text/{speaker_id}"] = speaker_text
        return outputs

    pipe.postprocess = MethodType(safe_postprocess, pipe)
    pipe._codex_safe_postprocess = True
    return pipe


def _normalize_segments(raw_result: dict[str, Any]) -> list[dict[str, Any]]:
    raw_chunks = raw_result.get("chunks")
    if not isinstance(raw_chunks, list):
        return []

    segments: list[dict[str, Any]] = []
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        timestamp = chunk.get("timestamp") or [0.0, 0.0]
        start = _safe_float(timestamp[0] if len(timestamp) > 0 else 0.0)
        end = _safe_float(timestamp[1] if len(timestamp) > 1 else start, start)
        if end < start:
            end = start
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "speaker": str(chunk.get("speaker_id") or "UNKNOWN"),
                "text": text,
            }
        )

    segments.sort(key=lambda item: (item["start"], item["speaker"]))
    return segments


def _collapse_segments(segments: list[dict[str, Any]], merge_gap_s: float = DEFAULT_MERGE_GAP_S) -> list[dict[str, Any]]:
    if not segments:
        return []

    collapsed: list[dict[str, Any]] = []
    for segment in segments:
        if not collapsed:
            collapsed.append(dict(segment))
            continue

        previous = collapsed[-1]
        same_speaker = previous["speaker"] == segment["speaker"]
        gap = float(segment["start"]) - float(previous["end"])
        if same_speaker and gap <= merge_gap_s:
            previous["end"] = max(float(previous["end"]), float(segment["end"]))
            previous["text"] = f"{previous['text']}{segment['text']}"
        else:
            collapsed.append(dict(segment))

    for item in collapsed:
        item["start"] = round(float(item["start"]), 2)
        item["end"] = round(float(item["end"]), 2)
    return collapsed


def _build_generate_kwargs(pipe: Any, prompt: Optional[str]) -> dict[str, Any]:
    generate_kwargs: dict[str, Any] = {
        "language": "ja",
        "task": "transcribe",
    }
    if not prompt:
        return generate_kwargs

    tokenizer = getattr(pipe, "tokenizer", None)
    get_prompt_ids = getattr(tokenizer, "get_prompt_ids", None)
    if not callable(get_prompt_ids):
        _log("tokenizer.get_prompt_ids() が見つからないため prompt は無視します", "warning")
        return generate_kwargs

    prompt_ids = get_prompt_ids(prompt, return_tensors="pt")
    model = getattr(pipe, "model", None)
    model_device = getattr(model, "device", None)
    if hasattr(prompt_ids, "to") and model_device is not None:
        prompt_ids = prompt_ids.to(model_device)
    generate_kwargs["prompt_ids"] = prompt_ids

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None and hasattr(prompt_ids, "ne"):
        generate_kwargs["attention_mask"] = prompt_ids.ne(pad_token_id)
    return generate_kwargs


def _punctuate_segments(segments: list[dict[str, Any]], pipe: Any, enabled: bool) -> list[dict[str, Any]]:
    if not enabled:
        return segments

    punctuator = getattr(pipe, "punctuator", None)
    punctuate = getattr(punctuator, "punctuate", None)
    if not callable(punctuate):
        return segments

    punctuated: list[dict[str, Any]] = []
    for segment in segments:
        item = dict(segment)
        try:
            item["text"] = str(punctuate(item["text"])).strip()
        except Exception as exc:
            _log(f"句読点付与に失敗したため元テキストを使用します: {exc}", "warning")
        punctuated.append(item)
    return punctuated


def _speaker_texts(raw_result: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, str]:
    speaker_ids = raw_result.get("speaker_ids")
    ordered_speakers: list[str] = []
    if isinstance(speaker_ids, list):
        ordered_speakers.extend(str(item) for item in speaker_ids)
    for segment in segments:
        speaker = str(segment["speaker"])
        if speaker not in ordered_speakers:
            ordered_speakers.append(speaker)

    texts: dict[str, str] = {}
    for speaker in ordered_speakers:
        value = raw_result.get(f"text/{speaker}")
        if value:
            texts[speaker] = str(value).strip()
            continue
        texts[speaker] = "".join(
            segment["text"] for segment in segments if str(segment["speaker"]) == speaker
        ).strip()
    return texts


def _render_output_text(segments: list[dict[str, Any]], header: Optional[str]) -> str:
    body = "\n".join(
        f"{segment['speaker']}: {segment['text']}"
        for segment in segments
        if str(segment.get("text") or "").strip()
    ).strip()

    if header:
        header_text = str(header).replace("\\n", "\n").strip()
        if body:
            return f"{header_text}\n\n{body}"
        return header_text
    return body


def _write_metadata(output_path: Path, metadata: dict[str, Any]) -> None:
    metadata_path = _metadata_path(output_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def transcribe_long_audio(
    audio_file: str,
    prompt: Optional[str] = None,
    header: Optional[str] = None,
    output_path: Optional[str] = None,
    vad_profile: str = "auto",
    **kwargs: Any,
) -> str:
    input_path = Path(audio_file)
    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {audio_file}")

    if vad_profile and vad_profile != "auto":
        _log(f"vad_profile={vad_profile} は kotoba v2.2 pipeline では使用しません", "debug")

    pipe = _get_or_create_pipeline(**kwargs)
    config = _resolve_pipeline_config(**kwargs)
    generate_kwargs = _build_generate_kwargs(pipe, prompt)

    inference_kwargs = {
        "chunk_length_s": config["chunk_length_s"],
        "add_punctuation": config["add_punctuation"],
        "generate_kwargs": generate_kwargs,
    }
    if config["stride_length_s"] is not None:
        inference_kwargs["stride_length_s"] = config["stride_length_s"]
    if config["num_speakers"] is not None:
        inference_kwargs["num_speakers"] = config["num_speakers"]
    if config["min_speakers"] is not None:
        inference_kwargs["min_speakers"] = config["min_speakers"]
    if config["max_speakers"] is not None:
        inference_kwargs["max_speakers"] = config["max_speakers"]
    if config["add_silence_start"] is not None:
        inference_kwargs["add_silence_start"] = config["add_silence_start"]
    if config["add_silence_end"] is not None:
        inference_kwargs["add_silence_end"] = config["add_silence_end"]

    _log(f"文字起こし開始: {input_path.name}")
    raw_result = pipe(str(input_path), **inference_kwargs)
    if not isinstance(raw_result, dict):
        raise RuntimeError("kotoba-whisper pipeline が想定外の形式を返しました")

    segments = _collapse_segments(_normalize_segments(raw_result))
    segments = _punctuate_segments(segments, pipe, enabled=config["add_punctuation"])
    speaker_texts = _speaker_texts(raw_result, segments)

    rendered_text = _render_output_text(segments, header=header)
    target_path = Path(output_path) if output_path else input_path.with_suffix(".txt")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(rendered_text, encoding="utf-8")

    metadata = {
        "text": rendered_text,
        "segments": segments,
        "language": "ja",
        "engine": ENGINE_ID,
        "model": config["model_id"],
        "speakers": [
            {
                "id": speaker_id,
                "text": speaker_texts[speaker_id],
            }
            for speaker_id in speaker_texts
        ],
        "source_audio": str(input_path.resolve()),
        "output_text": str(target_path.resolve()),
        "options": {
            "chunk_length_s": config["chunk_length_s"],
            "batch_size": config["batch_size"],
            "add_punctuation": config["add_punctuation"],
            "num_speakers": config["num_speakers"],
            "min_speakers": config["min_speakers"],
            "max_speakers": config["max_speakers"],
            "add_silence_start": config["add_silence_start"],
            "add_silence_end": config["add_silence_end"],
        },
    }
    _write_metadata(target_path, metadata)
    _log(f"文字起こし完了: {target_path.name}")
    return str(target_path)


def transcribe_file(
    audio_file: str,
    output_path: str,
    prompt: Optional[str] = None,
    header: Optional[str] = None,
    vad_profile: Optional[str] = None,
    **kwargs: Any,
) -> str:
    return transcribe_long_audio(
        audio_file=audio_file,
        output_path=output_path,
        prompt=prompt,
        header=header,
        vad_profile=vad_profile or "auto",
        **kwargs,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="kotoba-whisper-v2.2 で話者分離・句読点付き文字起こしを行います。",
    )
    parser.add_argument("audio_files", nargs="+", help="音声または動画ファイル")
    parser.add_argument("--prompt", "-p", default=None, help="Whisper prompt")
    parser.add_argument("--header", "-H", default=None, help="出力ファイル先頭に付与するヘッダー")
    parser.add_argument("--output", "-o", default=None, help="出力テキストファイル（単一入力のみ）")
    parser.add_argument("--chunk-length-s", type=int, default=None, help="ASR chunk length")
    parser.add_argument("--stride-length-s", type=float, default=None, help="ASR stride length")
    parser.add_argument("--batch-size", type=int, default=None, help="pipeline batch size")
    parser.add_argument("--num-speakers", type=int, default=None, help="話者数を固定")
    parser.add_argument("--min-speakers", type=int, default=None, help="最小話者数")
    parser.add_argument("--max-speakers", type=int, default=None, help="最大話者数")
    parser.add_argument("--add-silence-start", type=float, default=None, help="各話者区間の先頭に足す無音秒数")
    parser.add_argument("--add-silence-end", type=float, default=None, help="各話者区間の末尾に足す無音秒数")
    parser.add_argument("--no-punctuation", action="store_true", help="句読点後処理を無効化")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    _configure_stdio_for_windows()
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.output and len(args.audio_files) != 1:
        parser.error("--output は単一ファイル入力時のみ使用できます")

    exit_code = 0
    for index, audio_file in enumerate(args.audio_files):
        output_path = args.output if index == 0 else None
        try:
            transcribe_long_audio(
                audio_file=audio_file,
                prompt=args.prompt,
                header=args.header,
                output_path=output_path,
                chunk_length_s=args.chunk_length_s,
                stride_length_s=args.stride_length_s,
                batch_size=args.batch_size,
                add_punctuation=not args.no_punctuation,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                add_silence_start=args.add_silence_start,
                add_silence_end=args.add_silence_end,
            )
        except Exception as exc:
            exit_code = 1
            _log(f"処理失敗: {audio_file} - {exc}", "error")
    return exit_code
