#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kotoba-whisper v2.2 transcription entry points."""

from kotoba_transcriber.engine import (
    MODEL_ID,
    main,
    offload_model,
    restore_model,
    set_logger,
    transcribe_file,
    transcribe_long_audio,
)

__all__ = [
    "MODEL_ID",
    "main",
    "offload_model",
    "restore_model",
    "set_logger",
    "transcribe_file",
    "transcribe_long_audio",
]
