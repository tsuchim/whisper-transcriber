#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common adapter entry point for live-scraping style integration."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Optional


_WHISPER_MODULE: Optional[ModuleType] = None


def _load_whisper_module() -> ModuleType:
    global _WHISPER_MODULE
    if _WHISPER_MODULE is not None:
        return _WHISPER_MODULE

    script_path = Path(__file__).with_name("whisper-v3.py")
    spec = importlib.util.spec_from_file_location("whisper_v3_adapter_target", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"whisper-v3.py をロードできません: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _WHISPER_MODULE = module
    return module


def set_logger(logger) -> None:
    module = _load_whisper_module()
    if hasattr(module, "set_logger"):
        module.set_logger(logger)


def transcribe_file(
    audio_file: str,
    output_path: str,
    prompt: str | None = None,
    header: str | None = None,
    vad_profile: str | None = None,
    **kwargs,
) -> str:
    module = _load_whisper_module()
    module.transcribe_long_audio(
        audio_file=audio_file,
        prompt=prompt,
        header=header,
        output_path=output_path,
        vad_profile=vad_profile or "auto",
    )
    return output_path


def offload_model() -> None:
    module = _load_whisper_module()
    if hasattr(module, "offload_model"):
        module.offload_model()


def restore_model() -> None:
    module = _load_whisper_module()
    if hasattr(module, "restore_model"):
        module.restore_model()

