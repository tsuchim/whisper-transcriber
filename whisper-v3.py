#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2025 tsuchim

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import os
import subprocess
import shutil
import logging
from pathlib import Path
from typing import cast, Optional
import signal
import gc
import io
import argparse
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock

def _configure_stdio_for_windows() -> None:
    """Windows cp932 エンコーディングエラー回避（CLI実行時のみ）。"""
    if sys.platform != "win32":
        return
    # 既に StringIO 等に差し替えられているケースもあるため buffer の存在を確認する
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        # import 先（例: サービス/テスト）に副作用を持ち込まないため握りつぶす
        pass

# ロガー設定: 呼び出し元から渡されたロガーを使用、なければ標準出力
_logger: Optional[logging.Logger] = None
_logger_lock = Lock()

_models_init_lock = Lock()
_vad_init_lock = Lock()

def _log(msg: str, level: str = "info"):
    """ログ出力（logger があればそちらに、なければ print）"""
    with _logger_lock:
        logger = _logger

    if logger is not None:
        method = getattr(logger, level, None)
        if callable(method):
            method(msg)
        else:
            logger.info(msg)
        return

    # import 先でのログスパムを避けるため、logger 未設定時の debug は抑制
    if level == "debug":
        return
    print(msg)

def set_logger(logger: logging.Logger):
    """外部からロガーを設定"""
    global _logger
    with _logger_lock:
        _logger = logger

import librosa
import numpy as np
try:
    import av  # type: ignore

    has_pyav = True
except ImportError:
    av = None  # type: ignore
    has_pyav = False

try:
    import webrtcvad  # type: ignore

    has_webrtcvad = True
except Exception:
    webrtcvad = None  # type: ignore
    has_webrtcvad = False

def _av_open(path: str):
    """Open media via PyAV with conservative probe settings.

    Some containers (e.g. long FLV recordings) can intermittently fail stream
    detection/initial decode with default probe/analyze sizes. Use larger
    values to reduce false negatives.
    """
    # FFmpeg option values are strings.
    # - probesize: bytes
    # - analyzeduration: microseconds
    return av.open(  # type: ignore
        path,
        options={
            "probesize": "100M",
            "analyzeduration": "100000000",
        },
    )

try:
    import cupy as cp  # type: ignore

    has_cupy = True  # type: ignore
except ImportError:
    cp = np  # type: ignore
    has_cupy = False  # type: ignore
import torch
from pydub import AudioSegment  # type: ignore
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

has_silero_vad = False
_vad_model = None
_vad_utils = None
_get_speech_timestamps = None
detect_nonsilent = None

_FLOAT_EPS = np.finfo(np.float32).eps

# === Tuning constants (audio splitting / prefetch) ===
# VADで得られた区間の端に余白を足す (ms)
EDGE_PAD_MS = 500
# 長いセグメントは分割してWhisperへ渡す (ms)
LONG_SEG_THRESHOLD_MS = 20000
# 分割チャンク長 (ms)
SPLIT_CHUNK_MS = 15000
# 分割チャンクのスライド幅 (ms)
SPLIT_STEP_MS = 12000
# Whisperに渡す最小チャンク長 (ms)
MIN_CHUNK_MS = 500
# 小さな無音ギャップは結合する (ms)
JOIN_GAP_MS = 300
# Whisperに渡す最大チャンク長 (ms)
MAX_CHUNK_MS = 25000

# プリフェッチの結果待ちタイムアウト (秒)
PREFETCH_RESULT_TIMEOUT_SEC = 30.0


def _webrtcvad_ranges_ms(
    audio: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    padding_ms: int = 300,
    start_ratio: float = 0.9,
    end_ratio: float = 0.9,
) -> list[tuple[int, int]]:
    """WebRTC VAD で音声区間を検出（軽量・高速）。

    Returns:
        (start_ms, end_ms) のローカル範囲（chunk先頭=0）
    """
    if not has_webrtcvad or webrtcvad is None:
        raise RuntimeError("webrtcvad が利用できません")

    if sr not in {8000, 16000, 32000, 48000}:
        raise ValueError(f"webrtcvad unsupported sample rate: {sr}")
    if frame_ms not in {10, 20, 30}:
        raise ValueError(f"webrtcvad unsupported frame_ms: {frame_ms}")

    total_samples = int(audio.shape[0])
    if total_samples <= 0:
        return []

    # float32 [-1,1] → int16 PCM
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    bytes_per_frame = int(sr * frame_ms / 1000) * 2
    if bytes_per_frame <= 0:
        return []

    pcm_bytes = pcm.tobytes()
    n_frames = len(pcm_bytes) // bytes_per_frame
    if n_frames <= 0:
        return []

    vad = webrtcvad.Vad(int(aggressiveness))  # type: ignore[union-attr]
    padding_frames = max(1, int(round(padding_ms / frame_ms)))
    start_trigger = int(np.ceil(padding_frames * start_ratio))
    end_trigger = int(np.ceil(padding_frames * end_ratio))

    ring: list[bool] = []
    segments: list[tuple[int, int]] = []
    in_segment = False
    seg_start_frame = 0

    for i in range(n_frames):
        frame = pcm_bytes[i * bytes_per_frame : (i + 1) * bytes_per_frame]
        is_speech = vad.is_speech(frame, sr)  # type: ignore[union-attr]

        ring.append(bool(is_speech))
        if len(ring) > padding_frames:
            ring.pop(0)

        if not in_segment:
            if len(ring) == padding_frames and sum(ring) >= start_trigger:
                in_segment = True
                seg_start_frame = max(0, i - padding_frames + 1)
                ring.clear()
        else:
            if len(ring) == padding_frames and (padding_frames - sum(ring)) >= end_trigger:
                seg_end_frame = max(seg_start_frame, i - padding_frames + 1)
                start_ms = int(seg_start_frame * frame_ms)
                end_ms = int(seg_end_frame * frame_ms)
                if end_ms > start_ms:
                    segments.append((start_ms, end_ms))
                in_segment = False
                ring.clear()

    if in_segment:
        start_ms = int(seg_start_frame * frame_ms)
        end_ms = int(n_frames * frame_ms)
        if end_ms > start_ms:
            segments.append((start_ms, end_ms))

    return segments


def _ranges_intersection_ms(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """2つの (start_ms,end_ms) 範囲の交差長(ms)を計算する。"""
    if not a or not b:
        return 0
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    i = 0
    j = 0
    total = 0
    while i < len(a_sorted) and j < len(b_sorted):
        a_s, a_e = a_sorted[i]
        b_s, b_e = b_sorted[j]
        s = max(a_s, b_s)
        e = min(a_e, b_e)
        if e > s:
            total += e - s
        if a_e <= b_e:
            i += 1
        else:
            j += 1
    return total


def _ranges_total_ms(ranges: list[tuple[int, int]]) -> int:
    """範囲の合計長(ms)を計算（重複があればマージしてから合計）。"""
    merged = _merge_ranges_ms(ranges, gap_ms=0)
    return sum(max(0, e - s) for s, e in merged)


def _webrtcvad_calibrate_aggressiveness(audio: np.ndarray, sr: int, cap_sec: int = 3600) -> int:
    """WebRTC VAD の aggressiveness を安定性ベースで自動選択する。

    発話率などの「内容依存の閾値」に頼らず、aggressiveness=1..3 の結果の
    一致度（交差/和集合）から最も安定している設定を選ぶ。
    """
    if not has_webrtcvad:
        return 2
    if audio.size <= 0:
        return 2

    max_samples = int(sr * max(1, cap_sec))
    sample = audio[: min(int(audio.shape[0]), max_samples)]
    if sample.size <= 0:
        return 2

    results: dict[int, list[tuple[int, int]]] = {}
    for aggr in (1, 2, 3):
        results[aggr] = _webrtcvad_ranges_ms(sample, sr=sr, aggressiveness=aggr, frame_ms=30, padding_ms=300)

    scores: dict[int, float] = {}
    for aggr in (1, 2, 3):
        ratios: list[float] = []
        for other in (1, 2, 3):
            if other == aggr:
                continue
            inter = _ranges_intersection_ms(results[aggr], results[other])
            union = _ranges_total_ms(results[aggr]) + _ranges_total_ms(results[other]) - inter
            ratios.append(1.0 if union <= 0 else inter / union)
        scores[aggr] = sum(ratios) / max(1, len(ratios))

    best = max(scores.items(), key=lambda kv: kv[1])[0]
    # タイは中庸(2)を優先
    if scores.get(2, 0.0) == scores.get(best, 0.0):
        return 2
    return int(best)


def _webrtcvad_in_chunks(
    audio: np.ndarray,
    sr: int,
    chunk_count: int = 5,
    overlap_ms: int = 300,
    aggressiveness: Optional[int] = None,
) -> list[tuple[int, int]]:
    """WebRTC VAD を分割実行し、進捗ログを出力する。"""
    if not has_webrtcvad:
        raise RuntimeError("webrtcvad が利用できません")

    chosen_aggr = aggressiveness if aggressiveness is not None else _webrtcvad_calibrate_aggressiveness(audio, sr=sr)
    _log(f"WebRTC VAD aggressiveness(auto)={chosen_aggr}", "debug")

    total_samples = int(audio.shape[0])
    if total_samples <= 0:
        return []

    if chunk_count <= 1 or total_samples < sr * 60:
        chunk_count = 1

    is_single_chunk = chunk_count == 1
    overlap_samples = int(overlap_ms * sr / 1000)
    chunk_len = int(np.ceil(total_samples / chunk_count))

    all_ranges: list[tuple[int, int]] = []
    for i in range(chunk_count):
        base_start = i * chunk_len
        base_end = min(total_samples, (i + 1) * chunk_len)
        if base_start >= base_end:
            break

        chunk_start = max(0, base_start - overlap_samples)
        chunk_end = min(total_samples, base_end + overlap_samples)
        chunk_audio = audio[chunk_start:chunk_end]

        local = _webrtcvad_ranges_ms(
            chunk_audio,
            sr=sr,
            aggressiveness=chosen_aggr,
            frame_ms=30,
            padding_ms=300,
        )
        offset_ms = int(chunk_start * 1000 / sr)
        for s_ms, e_ms in local:
            start_ms = int(s_ms) + offset_ms
            end_ms = int(e_ms) + offset_ms
            if end_ms > start_ms:
                all_ranges.append((start_ms, end_ms))

        pct = int(round((i + 1) * 100 / chunk_count))
        if is_single_chunk:
            pct = 100
        _log(f"[PROGRESS] VAD進捗: {pct}% ({i+1}/{chunk_count})")

    return _merge_ranges_ms(all_ranges, gap_ms=150)


def init_vad() -> bool:
    """Silero VAD を初期化（初回のみ、必要になった時だけ）。"""
    global has_silero_vad, _vad_model, _vad_utils, _get_speech_timestamps, detect_nonsilent

    with _vad_init_lock:
        # pydub は常に利用可能にしておく（フォールバック用）
        if detect_nonsilent is None:
            try:
                from pydub.silence import detect_nonsilent as _detect_nonsilent  # type: ignore
                detect_nonsilent = _detect_nonsilent
            except Exception:
                detect_nonsilent = None

        if has_silero_vad and _vad_model is not None and _get_speech_timestamps is not None:
            return True

        try:
            _vad_model, _vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
                verbose=False,
            )
            _get_speech_timestamps, _, _, _, _ = _vad_utils
            
            # デバイス移動
            if torch.cuda.is_available():
                _vad_model.to(device)
                
            has_silero_vad = True
            _log("Silero VAD 初期化完了（高速音声検出）")
            return True
        except Exception as e:
            has_silero_vad = False
            _vad_model = None
            _vad_utils = None
            _get_speech_timestamps = None
            _log(f"Silero VAD 初期化失敗: {e}（pydub フォールバック）", "warning")
            return False


def _merge_ranges_ms(ranges: list[tuple[int, int]], gap_ms: int = 100) -> list[tuple[int, int]]:
    """(start_ms, end_ms) の範囲リストをソートしてマージする。"""
    if not ranges:
        return []

    ranges_sorted = sorted(ranges, key=lambda x: (x[0], x[1]))
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = ranges_sorted[0]
    for s, e in ranges_sorted[1:]:
        if s <= cur_e + gap_ms:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _silero_vad_in_chunks(
    audio: np.ndarray,
    sr: int,
    chunk_count: int = 5,
    overlap_ms: int = 300,
) -> list[tuple[int, int]]:
    """Silero VAD を分割実行し、進捗ログを出力する。

    注意:
      - 境界の取りこぼしを避けるため overlap を入れてからマージする。
      - 返り値は全体の (start_ms, end_ms) に正規化済み。
    """
    if not has_silero_vad or _vad_model is None or _get_speech_timestamps is None:
        raise RuntimeError("Silero VAD が利用できません")

    total_samples = int(audio.shape[0])
    if total_samples <= 0:
        return []

    # 短い音声は1チャンクで十分
    if chunk_count <= 1 or total_samples < sr * 60:
        chunk_count = 1

    is_single_chunk = chunk_count == 1

    overlap_samples = int(overlap_ms * sr / 1000)
    chunk_len = int(np.ceil(total_samples / chunk_count))

    all_ranges: list[tuple[int, int]] = []

    for i in range(chunk_count):
        base_start = i * chunk_len
        base_end = min(total_samples, (i + 1) * chunk_len)
        if base_start >= base_end:
            break

        # overlap 付きで切り出し
        chunk_start = max(0, base_start - overlap_samples)
        chunk_end = min(total_samples, base_end + overlap_samples)
        chunk_audio = audio[chunk_start:chunk_end]

        # VAD 実行
        audio_tensor = torch.from_numpy(chunk_audio).float()
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.to(device)

        # 推論専用: inference_mode の方が no_grad より高速/省メモリ
        with torch.inference_mode():
            speech_timestamps = _get_speech_timestamps(  # type: ignore[misc]
                audio_tensor,
                _vad_model,
                threshold=0.3,
                min_speech_duration_ms=300,
                min_silence_duration_ms=150,
                speech_pad_ms=100,
                return_seconds=False,
            )

        # サンプル→ms、全体座標へ変換
        for ts in speech_timestamps:
            start_samp = int(ts["start"]) + chunk_start
            end_samp = int(ts["end"]) + chunk_start
            start_ms = int(start_samp * 1000 / sr)
            end_ms = int(end_samp * 1000 / sr)
            if end_ms > start_ms:
                all_ranges.append((start_ms, end_ms))

        # GPU メモリ解放
        del speech_timestamps
        del audio_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 素直に進捗を出す（チャンク完了数 / 総数）
        pct = int(round((i + 1) * 100 / chunk_count))
        if is_single_chunk:
            pct = 100
        _log(f"[PROGRESS] VAD進捗: {pct}% ({i+1}/{chunk_count})")

    return _merge_ranges_ms(all_ranges, gap_ms=150)


def _pydub_vad_in_chunks(
    audio: np.ndarray,
    sr: int,
    chunk_count: int = 5,
    overlap_ms: int = 300,
    min_silence_len: int = 400,
    silence_thresh: int = -50,
    seek_step: int = 25,
) -> list[tuple[int, int]]:
    """pydub の detect_nonsilent を分割実行し、進捗ログを出力する。"""
    if detect_nonsilent is None:
        raise RuntimeError("pydub detect_nonsilent が利用できません")

    total_samples = int(audio.shape[0])
    if total_samples <= 0:
        return []

    if chunk_count <= 1 or total_samples < sr * 60:
        chunk_count = 1

    is_single_chunk = chunk_count == 1

    overlap_samples = int(overlap_ms * sr / 1000)
    chunk_len = int(np.ceil(total_samples / chunk_count))

    all_ranges: list[tuple[int, int]] = []

    for i in range(chunk_count):
        base_start = i * chunk_len
        base_end = min(total_samples, (i + 1) * chunk_len)
        if base_start >= base_end:
            break

        chunk_start = max(0, base_start - overlap_samples)
        chunk_end = min(total_samples, base_end + overlap_samples)
        chunk_audio = audio[chunk_start:chunk_end]

        audio_int16 = (chunk_audio * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1,
        )

        # pydub は chunk 内の ms を返すので、全体 ms に変換
        offset_ms = int(chunk_start * 1000 / sr)
        local_ranges = detect_nonsilent(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            seek_step=seek_step,
        )
        for s_ms, e_ms in local_ranges:
            start_ms = int(s_ms) + offset_ms
            end_ms = int(e_ms) + offset_ms
            if end_ms > start_ms:
                all_ranges.append((start_ms, end_ms))

        del audio_segment, audio_int16

        # 素直に進捗を出す（チャンク完了数 / 総数）
        pct = int(round((i + 1) * 100 / chunk_count))
        if is_single_chunk:
            pct = 100
        _log(f"[PROGRESS] VAD進捗: {pct}% ({i+1}/{chunk_count})")

    return _merge_ranges_ms(all_ranges, gap_ms=150)

model_id = "kotoba-tech/kotoba-whisper-v2.2"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = None
processor = None


def init_models() -> None:
    """Whisper（とVAD）を初期化（import時に重い処理をしないため遅延ロード）。"""
    global model, processor, _CLEANED
    with _models_init_lock:
        if model is not None and processor is not None:
            return

        _log(f"GPU利用可能: {torch.cuda.is_available()}", "debug")
        _log(f"transformers device: {device}", "debug")
        if has_cupy and torch.cuda.is_available():
            _log("CuPy GPU acceleration: 有効", "debug")
        else:
            _log("CuPy GPU acceleration: 無効 (CPU処理)", "debug")

        # Whisper モデル/プロセッサ初期化
        _log("Whisper モデルをロード中...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(  # type: ignore
            model_id,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)  # type: ignore
        _log("Whisper モデルロード完了")

        # VADは必要になったら init_vad() が呼ばれる（ここでは強制しない）

# === リソースクリーンアップ関連 ===
_CLEANED = False

def _cleanup_resources():
    """GPU/メモリ/モデルリソースを確実に解放する。
    強制終了（SIGINT/SIGTERM）や通常終了の両方で呼ばれる。
    """
    global model, processor, _vad_model, _CLEANED
    if _CLEANED:
        return
    _log("[CLEANUP] 開始: リソース解放処理")
    # 推論モデル削除
    try:
        if model is not None:
            del model
    except Exception:
        # 終了時のクリーンアップで発生した例外は無視し、呼び出し元の例外伝播を優先する
        pass
    try:
        if processor is not None:
            del processor
    except Exception:
        # 終了時のクリーンアップで発生した例外は無視し、呼び出し元の例外伝播を優先する
        pass
    # VAD モデル削除
    try:
        if _vad_model is not None:
            del _vad_model
    except Exception:
        # 終了時のクリーンアップで発生した例外は無視し、呼び出し元の例外伝播を優先する
        pass
    _vad_model = None
    model = None
    processor = None
    # CuPy メモリプール解放
    try:
        if has_cupy and hasattr(cp, "get_default_memory_pool"):
            cp.get_default_memory_pool().free_all_blocks()  # type: ignore
            _log("[CLEANUP] CuPy メモリプール解放")
    except Exception as e:
        _log(f"[CLEANUP] CuPy 解放失敗: {e}", "warning")
    # Torch GPU キャッシュ解放
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _log("[CLEANUP] torch.cuda.empty_cache 実行")
    except Exception as e:
        _log(f"[CLEANUP] torch キャッシュ解放失敗: {e}", "warning")
    # ガベージコレクション明示実行
    try:
        gc.collect()
        _log("[CLEANUP] gc.collect 実行")
    except Exception:
        # gc.collect が失敗しても致命的ではないため無視する
        pass
    _CLEANED = True
    _log("[CLEANUP] 完了")


def offload_model():
    """モデルをCPUに退避し、VRAMを解放する（他のアプリにGPUを譲るため）"""
    global model, _vad_model
    try:
        if model is not None:
            _log("[RESOURCE] Whisper モデルをCPUに退避中...", "debug")
            model.to("cpu")
    except Exception as e:
        _log(f"Whisperモデル退避失敗: {e}", "warning")

    try:
        if _vad_model is not None:
            _log("[RESOURCE] Silero VAD をCPUに退避中...", "debug")
            _vad_model.to("cpu")
    except Exception:
        # 一部環境で VAD モデルのデバイス移動に失敗する場合があるため無視する
        pass
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    _log("[RESOURCE] VRAM解放完了 (CPU待機状態)", "debug")


def restore_model():
    """モデルをGPUに復帰させる（処理開始前）"""
    global model, _vad_model
    if model is None or processor is None:
        init_models()
    if not torch.cuda.is_available():
        return

    # 既にGPUにあるか確認（簡易チェック）
    try:
        if model is not None and next(model.parameters()).device.type == "cuda":
            return
    except Exception:
        # 状態確認に失敗しても致命的ではないため継続する
        pass

    try:
        if model is not None:
            _log("[RESOURCE] Whisper モデルをGPUに復帰中...", "debug")
            model.to(device)
    except Exception as e:
        _log(f"Whisperモデル復帰失敗: {e}", "warning")

    try:
        if _vad_model is not None:
            _log("[RESOURCE] Silero VAD をGPUに復帰中...", "debug")
            _vad_model.to(device)
    except Exception:
        # 一部環境で VAD モデルのデバイス移動に失敗する場合があるため無視する
        pass
    _log("[RESOURCE] GPU復帰完了", "debug")


def _signal_handler(signum, frame):
    """SIGINT/SIGTERM を捕捉して安全に終了"""
    _log(f"[CLEANUP] シグナル受信: {signum}. クリーンアップを実行して終了します", "warning")
    _cleanup_resources()
    # シグナルコードを反映した終了ステータス (POSIX 慣例)
    try:
        sys.exit(128 + signum)
    except SystemExit:
        raise

def _register_signal_handlers() -> None:
    # import 先へ副作用を持ち込まないため、CLI実行時にのみ呼ぶ
    try:
        signal.signal(signal.SIGINT, _signal_handler)
    except Exception:
        # 一部の環境（例: 制限付きランタイムやOS）ではシグナルハンドラの登録に失敗するため、
        # その場合は処理継続を優先し、失敗を無視する。
        pass
    try:
        signal.signal(signal.SIGTERM, _signal_handler)  # type: ignore
    except Exception:
        # 上記と同様に、シグナル登録に失敗しても致命的ではないため無視する。
        pass


def _ffmpeg_available() -> bool:
    """ffmpegがPATHにあるかを確認"""
    return shutil.which("ffmpeg") is not None


def load_audio_via_ffmpeg(path: str, sr: int = 16000) -> np.ndarray:
    """
    ffmpegで動画コンテナから音声ストリームのみを抽出し、16kHz/mono/PCMで読み込む。
    -vn でビデオ無効化、-ac 1 でモノラル、-ar でサンプリング周波数、-f s16le で16bit PCM。
    """
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "s16le",
        "-",
    ]
    startupinfo = None
    creationflags = 0
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        creationflags = subprocess.CREATE_NO_WINDOW
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        startupinfo=startupinfo,
        creationflags=creationflags,
    )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm.size == 0:
        raise ValueError("ffmpegで音声データを取得できませんでした")
    audio = pcm.astype(np.float32) / 32767.0
    return audio


def load_audio_chunk_via_ffmpeg(path: str, start_sec: float, duration_sec: float, sr: int = 16000) -> np.ndarray:
    """
    ffmpegで指定した時間区間のみを音声として読み込む（ストリーミング方式）。
    
    Args:
        path: 音声/動画ファイルパス
        start_sec: 開始位置（秒）
        duration_sec: 読み込む長さ（秒）
        sr: サンプリングレート
    
    Returns:
        指定区間の音声データ（float32, -1.0〜1.0）
    """
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-ss", str(start_sec),      # シーク位置（入力の前に指定で高速）
        "-i", path,
        "-t", str(duration_sec),    # 読み込む長さ
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "s16le",
        "-",
    ]
    startupinfo = None
    creationflags = 0
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        creationflags = subprocess.CREATE_NO_WINDOW
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        startupinfo=startupinfo,
        creationflags=creationflags,
    )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm.size == 0:
        return np.array([], dtype=np.float32)
    audio = pcm.astype(np.float32) / 32767.0
    return audio


def load_audio_chunk_via_pyav(path: str, start_sec: float, duration_sec: float, sr: int = 16000) -> np.ndarray:
    """
    PyAVで指定した時間区間のみを音声として読み込む（ストリーミング方式）。
    
    Args:
        path: 音声/動画ファイルパス
        start_sec: 開始位置（秒）
        duration_sec: 読み込む長さ（秒）
        sr: サンプリングレート
    
    Returns:
        指定区間の音声データ（float32, -1.0〜1.0）
    """
    if not has_pyav:
        raise RuntimeError("PyAVがインストールされていません")
    
    end_sec = start_sec + duration_sec
    
    with _av_open(path) as container:  # type: ignore
        audio_stream = None
        for s in container.streams:  # type: ignore
            if s.type == "audio":  # type: ignore
                audio_stream = s
                break
        if audio_stream is None:
            raise ValueError("音声ストリームが見つかりませんでした")

        # シーク（キーフレーム単位なので多少前から始まる可能性あり）
        start_pts = int(start_sec / audio_stream.time_base)  # type: ignore
        container.seek(start_pts, stream=audio_stream)  # type: ignore

        # NOTE: Do not force output channel layout here. Some streams/container
        # combinations can raise IndexError internally when layout is set.
        # We downmix to mono ourselves from ndarray.
        resampler = av.audio.resampler.AudioResampler(  # type: ignore
            format="s16",
            rate=sr,
        )

        pcm_chunks: list[np.ndarray] = []
        samples_needed = int(duration_sec * sr)
        samples_collected = 0
        
        for packet in container.demux(audio_stream):  # type: ignore
            for frame in packet.decode():  # type: ignore
                # フレームの時刻を取得
                frame_time = float(frame.pts * audio_stream.time_base) if frame.pts is not None else 0  # type: ignore
                
                # 開始位置より前のフレームはスキップ
                if frame_time < start_sec - 0.1:  # 100msのマージン
                    continue
                # 終了位置を超えたら終了
                if frame_time > end_sec + 0.1:
                    break
                
                resampled = resampler.resample(frame)  # type: ignore
                if resampled is None:
                    continue

                # PyAV's AudioResampler.resample may return either a single AudioFrame
                # or a sequence (e.g., list/tuple) of AudioFrame objects, depending on buffering.
                frames = resampled if isinstance(resampled, (list, tuple)) else [resampled]
                for f in frames:
                    buf = f.to_ndarray()  # type: ignore
                    if buf.ndim == 1:
                        pcm = buf.astype(np.int16)
                    else:
                        pcm = buf.astype(np.int16).mean(axis=0).astype(np.int16)
                    if pcm.size:
                        pcm_chunks.append(pcm)
                        samples_collected += pcm.size
                
                if samples_collected >= samples_needed:
                    break
            
            if samples_collected >= samples_needed:
                break

        if not pcm_chunks:
            return np.array([], dtype=np.float32)

        pcm_all = np.concatenate(pcm_chunks)
        # 必要なサンプル数に切り詰め
        if len(pcm_all) > samples_needed:
            pcm_all = pcm_all[:samples_needed]
        audio = pcm_all.astype(np.float32) / 32767.0
        return audio


def load_audio_via_pyav(path: str, sr: int = 16000) -> np.ndarray:
    """
    PyAV(FFmpegバインディング)で動画から音声のみをデコードして16kHz/mono/PCMとして取得。
    外部プロセスを起動せずに、FFmpegのライブラリを経由してデコードします。
    """
    if not has_pyav:
        raise RuntimeError("PyAVがインストールされていません")
    # コンテナを開く
    with _av_open(path) as container:  # type: ignore
        # 最初の音声ストリームを選択
        audio_stream = None
        for s in container.streams:  # type: ignore
            if s.type == "audio":  # type: ignore
                audio_stream = s
                break
        if audio_stream is None:
            raise ValueError("音声ストリームが見つかりませんでした")

        # 16kHz/mono/16bitPCMへリサンプル
        # NOTE: Do not force output channel layout here. Some streams/container
        # combinations can raise IndexError internally when layout is set.
        # We downmix to mono ourselves from ndarray.
        resampler = av.audio.resampler.AudioResampler(  # type: ignore
            format="s16",
            rate=sr,
        )

        pcm_chunks: list[np.ndarray] = []
        for packet in container.demux(audio_stream):  # type: ignore
            for frame in packet.decode():  # type: ignore
                # resampleの戻りは None / AudioFrame / [AudioFrame, ...] の場合がある
                resampled = resampler.resample(frame)  # type: ignore

                if resampled is None:
                    continue

                # resampleの戻りは None / AudioFrame / [AudioFrame, ...] の場合があるため正規化する
                frames = resampled if isinstance(resampled, (list, tuple)) else [resampled]

                for f in frames:
                    # to_ndarray: shapeは(チャンネル, サンプル) もしくは (サンプル,)
                    buf = f.to_ndarray()  # type: ignore
                    if buf.ndim == 1:
                        pcm = buf.astype(np.int16)
                    else:
                        # 念のため多chの形でもサンプル次元に揃える
                        pcm = buf.astype(np.int16).mean(axis=0).astype(np.int16)
                    if pcm.size:
                        pcm_chunks.append(pcm)

        # フラッシュ: 残りのサンプルを取り出す
        try:
            flushed = resampler.resample(None)  # type: ignore
            if flushed is not None:
                frames = flushed if isinstance(flushed, (list, tuple)) else [flushed]
                for f in frames:
                    buf = f.to_ndarray()  # type: ignore
                    if buf.ndim == 1:
                        pcm = buf.astype(np.int16)
                    else:
                        pcm = buf.astype(np.int16).mean(axis=0).astype(np.int16)
                    if pcm.size:
                        pcm_chunks.append(pcm)
        except Exception:
            # 一部のPyAV環境ではflushが未対応/例外となる場合があるため無視する
            pass

        if not pcm_chunks:
            raise ValueError("PyAVで音声データを取得できませんでした")

        pcm_all = np.concatenate(pcm_chunks)
        audio = pcm_all.astype(np.float32) / 32767.0
        return audio


class InMemoryChunkProvider:
    """
    メモリ上の PCM データからチャンクを切り出すプロバイダ。
    
    VAD で使用した PCM データをそのまま再利用し、Whisper 推論時の
    ファイル I/O を完全に排除する。毎回ファイルをシークして読み込む
    オーバーヘッドがなくなり、大幅な高速化を実現する。
    
    メモリ使用量:
        - 16kHz mono float32 で 1時間 = 約 230MB
        - 2時間の配信でも 460MB 程度
        - Whisper モデル自体が数 GB 使用するため、相対的に小さい
    """
    
    def __init__(self, audio_data: np.ndarray, chunks: list, sr: int = 16000):
        """
        Args:
            audio_data: VAD で使用済みの PCM データ (float32, -1.0〜1.0)
            chunks: (start_ms, end_ms) のリスト
            sr: サンプリングレート（デフォルト 16000）
        """
        self.audio_data = audio_data
        self.chunks = chunks
        self.sr = sr
        self._mode = "in-memory (zero I/O)"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # audio_data の解放は呼び出し側に任せる
        return False
    
    def get_chunk(self, chunk_idx: int) -> Optional[np.ndarray]:
        """
        指定インデックスのチャンクをメモリからスライスして返す。
        
        Returns:
            チャンクの音声データ（float32）、または None
        """
        if chunk_idx >= len(self.chunks):
            return None
        
        start_ms, end_ms = self.chunks[chunk_idx]
        duration_ms = end_ms - start_ms
        
        if duration_ms < 200:
            return None  # 短すぎるチャンク
        
        # ミリ秒 → サンプルインデックスに変換
        start_sample = int(start_ms * self.sr / 1000)
        end_sample = int(end_ms * self.sr / 1000)
        
        # 配列範囲チェック
        if start_sample >= len(self.audio_data):
            return None
        end_sample = min(end_sample, len(self.audio_data))
        
        # スライス（コピーではなくビュー、ただし後で正規化するのでコピーが必要）
        chunk_data = self.audio_data[start_sample:end_sample].copy()
        
        if len(chunk_data) < 100:
            return None
        
        # 正規化（AudioChunkPrefetcher._load_chunk と同じ処理）
        chunk_data = np.nan_to_num(chunk_data, nan=0.0, posinf=1.0, neginf=-1.0)
        rms = np.sqrt(np.mean(chunk_data**2))
        if rms > 1e-6:
            target_rms = 0.1
            gain = min(target_rms / rms, 10.0)
            chunk_data = chunk_data * gain
            chunk_data = np.clip(chunk_data, -1.0, 1.0)
        
        return chunk_data
    
    def start_prefetch(self, start_idx: int):
        """互換性のためのダミーメソッド（メモリアクセスなのでプリフェッチ不要）"""
        pass
    
    @property
    def mode(self) -> str:
        """使用中のモード"""
        return self._mode


class AudioChunkPrefetcher:
    """
    非同期プリフェッチによる高速チャンク読み込み。
    GPU推論中に次のチャンクをバックグラウンドで読み込み、I/O待ちを隠蔽する。
    
    注意: InMemoryChunkProvider が利用可能な場合はそちらを優先使用すること。
    このクラスは audio_data がメモリに保持されていない場合のフォールバック用。
    """
    
    def __init__(
        self,
        audio_file: str,
        chunks: list,
        prefetch_count: int = 3,
        prefetch_timeout_sec: float = PREFETCH_RESULT_TIMEOUT_SEC,
    ):
        """
        Args:
            audio_file: 音声/動画ファイルパス
            chunks: (start_ms, end_ms) のリスト
            prefetch_count: 先読みするチャンク数（デフォルト3）
        """
        self.audio_file = audio_file
        self.chunks = chunks
        self.prefetch_count = prefetch_count
        self.prefetch_timeout_sec = float(prefetch_timeout_sec)
        
        # ストリーミング方式を判定
        ext = Path(audio_file).suffix.lower()
        is_video_container = ext in {".mkv", ".mp4", ".webm", ".mov", ".m4v", ".flv", ".avi", ".wmv"}
        self.use_pyav = is_video_container and has_pyav
        self.use_ffmpeg = is_video_container and _ffmpeg_available() and not self.use_pyav
        
        # PyAV用: コンテナを1回だけ開いて保持
        self._container = None
        self._audio_stream = None
        self._resampler = None
        self._container_lock = Lock()
        
        # プリフェッチ用スレッドプール
        self._executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_futures: dict[int, Future] = {}
        
        if self.use_pyav:
            self._mode = "PyAV (persistent container, no prefetch)"
        elif self.use_ffmpeg:
            self._mode = "ffmpeg (threaded prefetch)"
        else:
            self._mode = "librosa"
    
    def __enter__(self):
        """コンテキストマネージャ: 開始"""
        if self.use_pyav:
            self._open_pyav_container()
            # PyAV persistent container はシングルコンテナなのでプリフェッチ無効
            # マルチスレッドでのロック競合・タイムアウトを完全に回避
            self._executor = None
        else:
            # ffmpeg/librosa モードはプリフェッチ有効
            self._executor = ThreadPoolExecutor(max_workers=self.prefetch_count)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャ: 終了"""
        if self._executor:
            # 終了時にできるだけプリフェッチを完了させるが、無限待機は避ける
            try:
                from concurrent.futures import wait as _wait

                _wait(list(self._prefetch_futures.values()), timeout=2.0)
            except Exception:
                # cleanup時の例外は無視する（呼び出し元の例外を優先）
                pass
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        if self._container:
            try:
                self._container.close()
            except Exception:
                # cleanup時の例外は無視する（呼び出し元の例外を優先）
                pass
            self._container = None
        return False
    
    def _open_pyav_container(self):
        """PyAVコンテナを開く（1回のみ）"""
        if not has_pyav:
            return
        try:
            self._container = _av_open(self.audio_file)  # type: ignore
            for s in self._container.streams:  # type: ignore
                if s.type == "audio":  # type: ignore
                    self._audio_stream = s
                    break
            if self._audio_stream:
                self._resampler = av.audio.resampler.AudioResampler(  # type: ignore
                    format="s16",
                    rate=16000,
                )
        except Exception as e:
            _log(f"PyAVコンテナ初期化失敗: {e}", "warning")
            self._container = None
            self.use_pyav = False
            self.use_ffmpeg = _ffmpeg_available()
    
    def _load_chunk_pyav_persistent(self, start_sec: float, duration_sec: float) -> np.ndarray:
        """
        開いたままのPyAVコンテナからチャンクを読み込む（シーク＋デコード）。
        スレッドセーフのためロックを使用。
        """
        if not self._container or not self._audio_stream:
            return np.array([], dtype=np.float32)
        
        end_sec = start_sec + duration_sec
        
        with self._container_lock:
            try:
                # シーク
                start_pts = int(start_sec / self._audio_stream.time_base)  # type: ignore
                self._container.seek(start_pts, stream=self._audio_stream)  # type: ignore
                
                # resampler は基本再利用し、必要なら状態をリセットする
                resampler = self._resampler
                if resampler is None:
                    resampler = av.audio.resampler.AudioResampler(  # type: ignore
                        format="s16",
                        rate=16000,
                    )
                    self._resampler = resampler
                else:
                    try:
                        resampler.reset()  # type: ignore[attr-defined]
                    except AttributeError:
                        # reset() がない環境では安全のため再生成する
                        resampler = av.audio.resampler.AudioResampler(  # type: ignore
                            format="s16",
                            rate=16000,
                        )
                        self._resampler = resampler
                
                pcm_chunks: list[np.ndarray] = []
                samples_needed = int(duration_sec * 16000)
                samples_collected = 0
                
                for packet in self._container.demux(self._audio_stream):  # type: ignore
                    for frame in packet.decode():  # type: ignore
                        frame_time = float(frame.pts * self._audio_stream.time_base) if frame.pts is not None else 0  # type: ignore
                        
                        if frame_time < start_sec - 0.1:
                            continue
                        if frame_time > end_sec + 0.1:
                            break
                        
                        resampled = resampler.resample(frame)  # type: ignore
                        if resampled is None:
                            continue
                        
                        frames = resampled if isinstance(resampled, (list, tuple)) else [resampled]
                        for f in frames:
                            buf = f.to_ndarray()  # type: ignore
                            if buf.ndim == 1:
                                pcm = buf.astype(np.int16)
                            else:
                                pcm = buf.astype(np.int16).mean(axis=0).astype(np.int16)
                            if pcm.size:
                                pcm_chunks.append(pcm)
                                samples_collected += pcm.size
                        
                        if samples_collected >= samples_needed:
                            break
                    
                    if samples_collected >= samples_needed:
                        break
                
                if not pcm_chunks:
                    return np.array([], dtype=np.float32)
                
                pcm_all = np.concatenate(pcm_chunks)
                if len(pcm_all) > samples_needed:
                    pcm_all = pcm_all[:samples_needed]
                return pcm_all.astype(np.float32) / 32767.0
                
            except Exception as e:
                _log(f"PyAV persistent read error: {e}", "warning")
                return np.array([], dtype=np.float32)
    
    def _load_chunk(self, chunk_idx: int) -> tuple[int, Optional[np.ndarray]]:
        """
        指定インデックスのチャンクを読み込む（ワーカースレッド用）。
        
        Returns:
            (chunk_idx, audio_data or None)
        """
        if chunk_idx >= len(self.chunks):
            return (chunk_idx, None)
        
        start_ms, end_ms = self.chunks[chunk_idx]
        duration_ms = end_ms - start_ms
        
        if duration_ms < 200:
            return (chunk_idx, None)  # 短すぎるチャンク
        
        start_sec = start_ms / 1000.0
        duration_sec = duration_ms / 1000.0
        
        try:
            if self.use_pyav and self._container:
                chunk_data = self._load_chunk_pyav_persistent(start_sec, duration_sec)
            elif self.use_ffmpeg:
                chunk_data = load_audio_chunk_via_ffmpeg(self.audio_file, start_sec, duration_sec, sr=16000)
            else:
                chunk_data, _ = librosa.load(self.audio_file, sr=16000, offset=start_sec, duration=duration_sec)  # type: ignore
            
            if len(chunk_data) < 100:
                return (chunk_idx, None)
            
            # 正規化もここで実行（CPUバウンドだがGPUを待たせないため）
            chunk_data = np.nan_to_num(chunk_data, nan=0.0, posinf=1.0, neginf=-1.0)
            rms = np.sqrt(np.mean(chunk_data**2))
            if rms > 1e-6:
                target_rms = 0.1
                gain = min(target_rms / rms, 10.0)
                chunk_data = chunk_data * gain
                chunk_data = np.clip(chunk_data, -1.0, 1.0)
            
            return (chunk_idx, chunk_data)
            
        except Exception as e:
            error_msg = str(e) or repr(e) or type(e).__name__
            _log(f"Chunk {chunk_idx} load error [{type(e).__name__}]: {error_msg}", "warning")
            return (chunk_idx, None)
    
    def start_prefetch(self, start_idx: int):
        """指定インデックスから prefetch_count 個先までプリフェッチを開始"""
        # プリフェッチ無効（PyAV persistent mode）の場合はスキップ
        if not self._executor:
            return
        
        for i in range(start_idx, min(start_idx + self.prefetch_count, len(self.chunks))):
            if i not in self._prefetch_futures:
                self._prefetch_futures[i] = self._executor.submit(self._load_chunk, i)
    
    def get_chunk(self, chunk_idx: int) -> Optional[np.ndarray]:
        """
        チャンクを取得。プリフェッチ済みならそれを返し、なければ同期読み込み。
        次のチャンクのプリフェッチも開始する。
        """
        # プリフェッチ無効（PyAV persistent mode）の場合は直接同期読み込み
        if not self._executor:
            _, data = self._load_chunk(chunk_idx)
            return data
        
        # 次のプリフェッチを開始
        self.start_prefetch(chunk_idx + 1)
        
        # プリフェッチ結果を確認
        if chunk_idx in self._prefetch_futures:
            future = self._prefetch_futures.pop(chunk_idx)
            try:
                result = future.result(timeout=self.prefetch_timeout_sec)
                # _load_chunk は (idx, data) を返す。data が None なら読み込み失敗
                _, data = result
                if data is not None:
                    return data
                # data=None: プリフェッチでの読み込み失敗 → 警告のみ（再試行しない）
                _log(f"Prefetch chunk {chunk_idx} returned None data", "warning")
                return None
            except Exception as e:
                error_msg = str(e) or repr(e) or type(e).__name__
                _log(f"Prefetch result error for chunk {chunk_idx} [{type(e).__name__}]: {error_msg}", "warning")
                # 例外時はフォールバック同期読み込み
                _, data = self._load_chunk(chunk_idx)
                return data
        
        # プリフェッチされていなければ同期読み込み
        _, data = self._load_chunk(chunk_idx)
        return data
    
    @property
    def mode(self) -> str:
        """使用中のストリーミングモード"""
        return self._mode


def _parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を解析
    
    Returns:
        解析された引数のNamespace
    """
    parser = argparse.ArgumentParser(
        description='音声/動画ファイルを日本語で文字起こしします (kotoba-whisper-v2.2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  %(prog)s audio.mp3
  %(prog)s video.mp4 --prompt "配信者: 山田太郎。キーワード: ゲーム配信"
  %(prog)s audio.flv --header "配信: 山田太郎\\n日時: 2025-01-01 12:00" --output /path/to/output.txt
"""
    )
    parser.add_argument(
        'audio_files',
        nargs='+',
        help='文字起こしする音声/動画ファイル（複数指定可）'
    )
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default=None,
        help='AIに渡すプロンプト文字列（文脈情報、話者名など）'
    )
    parser.add_argument(
        '--header', '-H',
        type=str,
        default=None,
        help=r'出力ファイルの冒頭に付加するヘッダ文字列（\n で改行）'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='出力ファイルパス（単一ファイル処理時のみ有効）'
    )
    parser.add_argument(
        "--vad-profile",
        type=str,
        default="auto",
        choices=["auto", "fast", "accurate"],
        help="VADのプロファイル: fast=軽量優先, accurate=検出性能優先, auto=既定",
    )
    
    args = parser.parse_args()
    
    # --output は単一ファイル時のみ有効
    if args.output and len(args.audio_files) > 1:
        parser.error('--output は単一ファイル処理時のみ指定できます')
    
    return args


def transcribe_long_audio(
    audio_file: str,
    prompt: Optional[str] = None,
    header: Optional[str] = None,
    output_path: Optional[str] = None,
    vad_profile: str = "auto",
) -> str:
    """長い音声を分割して文字起こしし、結果を連結して返す。"""
    import time as _time

    if model is None or processor is None:
        init_models()

    audio_path_obj = Path(audio_file)
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"入力ファイルが存在しません: {audio_file}")

    _log(f"=== 文字起こし開始: {audio_path_obj.name} ===")
    if prompt:
        _log(f"プロンプト: {prompt}")

    try:
        ext = audio_path_obj.suffix.lower()
        is_video_container = ext in {".mkv", ".mp4", ".webm", ".mov", ".m4v", ".flv", ".avi", ".wmv"}
        if is_video_container and _ffmpeg_available():
            _log("ffmpeg経由で音声ストリームのみを読み込みます (-vn, 16kHz/mono)")
            audio = load_audio_via_ffmpeg(audio_file, sr=16000)
        else:
            audio, _ = librosa.load(audio_file, sr=16000)  # type: ignore
    except Exception as e:
        _log(f"音声読み込みエラー: {e}", "error")
        raise RuntimeError(f"音声読み込みに失敗しました: {audio_file}") from e

    audio_duration_sec = len(audio) / 16000
    _log(f"[PROGRESS] 音声読み込み完了: {audio_duration_sec:.1f}秒 ({audio_duration_sec/60:.1f}分)")
    _log(f"[PROGRESS] 前処理開始... (GPU: {torch.cuda.is_available()}, デバイス: {device})")
    _preprocess_start = _time.time()

    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        target_rms = 0.1
        audio = np.clip(audio * (target_rms / rms), -1.0, 1.0)
    audio = cast(np.ndarray, audio)

    audio_length_ms = int(audio_duration_sec * 1000)
    _vad_start = _time.time()
    vad_profile_norm = (vad_profile or "auto").strip().lower()
    vad_chunk_count = max(1, int(audio_duration_sec // 3600))

    def _run_vad_backend(name: str) -> Optional[list[tuple[int, int]]]:
        try:
            if name == "webrtcvad":
                if not has_webrtcvad:
                    raise RuntimeError("webrtcvad がインストールされていません")
                ranges = _webrtcvad_in_chunks(
                    audio,
                    sr=16000,
                    chunk_count=vad_chunk_count,
                    overlap_ms=300,
                    aggressiveness=None,
                )
                _log(f"[PROGRESS] WebRTC VAD 完了: {len(ranges)} セグメント検出, {_time.time() - _vad_start:.2f}秒")
                return ranges

            if name == "silero":
                init_vad()
                if not has_silero_vad:
                    raise RuntimeError("Silero VAD が利用できません")
                ranges = _silero_vad_in_chunks(audio, sr=16000, chunk_count=vad_chunk_count, overlap_ms=300)
                _log(f"[PROGRESS] Silero VAD 完了: {len(ranges)} セグメント検出, {_time.time() - _vad_start:.2f}秒")
                return ranges

            if name == "pydub":
                init_vad()
                ranges = _pydub_vad_in_chunks(
                    audio,
                    sr=16000,
                    chunk_count=vad_chunk_count,
                    overlap_ms=300,
                    min_silence_len=400,
                    silence_thresh=-50,
                    seek_step=25,
                )
                _log(f"[PROGRESS] pydub VAD 完了: {len(ranges)} セグメント検出, {_time.time() - _vad_start:.2f}秒")
                return ranges

            raise ValueError(f"unknown VAD backend: {name}")
        except Exception as e:
            # 「利用不可」「実行時エラー」どちらも同じくフォールバック対象
            _log(f"[PROGRESS] VAD失敗: {name} -> フォールバック ({type(e).__name__}: {e})", "warning")
            return None

    # profile に応じて優先順位を決める（常に失敗時は warning を出して次へ）
    if vad_profile_norm == "fast":
        backends = ["webrtcvad", "silero", "pydub"]
    elif vad_profile_norm == "accurate":
        backends = ["silero", "webrtcvad", "pydub"]
    else:
        # auto: 既定は精度優先（従来互換）
        backends = ["silero", "webrtcvad", "pydub"]

    nonsilent_ranges: list[tuple[int, int]] = []
    for b in backends:
        got = _run_vad_backend(b)
        if got is not None:
            nonsilent_ranges = got
            break

    if not nonsilent_ranges and audio_length_ms > 0:
        # VAD 全滅時でも処理は継続する（精度は落ちるが停止よりマシ）
        _log("[PROGRESS] 警告: 利用可能なVADが無いため全区間をWhisperへ渡します", "warning")
        nonsilent_ranges = [(0, audio_length_ms)]

    _log(f"[PROGRESS] VAD完了 (PCMデータ保持: {len(audio) * 4 / 1024 / 1024:.1f}MB)")

    if nonsilent_ranges:
        ranges_list = list(nonsilent_ranges)
        first_s, first_e = ranges_list[0]
        last_s, last_e = ranges_list[-1]
        ranges_list[0] = (max(0, first_s - EDGE_PAD_MS), first_e)
        if len(ranges_list) == 1:
            ranges_list[0] = (ranges_list[0][0], min(audio_length_ms, first_e + EDGE_PAD_MS))
        else:
            ranges_list[-1] = (last_s, min(audio_length_ms, last_e + EDGE_PAD_MS))
        nonsilent_ranges = ranges_list

    processed_ranges: list[tuple[int, int]] = []
    for start_ms, end_ms in nonsilent_ranges:
        duration = end_ms - start_ms
        if duration > LONG_SEG_THRESHOLD_MS:
            pos = start_ms
            while pos < end_ms:
                seg_end = min(pos + SPLIT_CHUNK_MS, end_ms)
                if seg_end - pos >= MIN_CHUNK_MS:
                    processed_ranges.append((pos, seg_end))
                pos += SPLIT_STEP_MS
        else:
            processed_ranges.append((start_ms, end_ms))

    chunks: list[tuple[int, int]] = []
    cur_start = -1
    cur_end = -1
    for start_ms, end_ms in processed_ranges:
        if cur_start < 0:
            cur_start, cur_end = start_ms, end_ms
            continue

        gap = start_ms - cur_end
        merged_end = max(cur_end, end_ms)
        merged_span = merged_end - cur_start
        if (gap <= JOIN_GAP_MS) and (merged_span <= MAX_CHUNK_MS):
            cur_end = merged_end
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = start_ms, end_ms
    if cur_start >= 0:
        chunks.append((cur_start, cur_end))

    _preprocess_time = _time.time() - _preprocess_start
    total_speech_sec = sum((e - s) / 1000 for s, e in chunks)
    _log(f"[PROGRESS] 前処理完了: {_preprocess_time:.2f}秒 / チャンク数: {len(chunks)} / 音声合計: {total_speech_sec:.1f}秒 / Whisper推論開始")
    _inference_start = _time.time()

    prompt_ids = None
    if prompt:
        try:
            prompt_ids = processor.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")  # type: ignore[union-attr]
            prompt_ids = prompt_ids.squeeze(0).to(device)
            _log(f"prompt_ids shape: {prompt_ids.shape}", "debug")  # type: ignore
        except Exception as e:
            _log(f"プロンプトトークン化エラー(無視して続行): {e}", "warning")
            prompt_ids = None

    transcriptions: list[str] = []
    chunk_errors = 0

    with InMemoryChunkProvider(audio, chunks) as chunk_provider:
        _log(f"[PROGRESS] チャンク読み込みモード: {chunk_provider.mode}")
        next_threshold = 20 if chunks else 100

        for i in range(len(chunks)):
            pct = (i + 1) * 100.0 / len(chunks)
            is_last = i == len(chunks) - 1
            if is_last or pct >= next_threshold:
                elapsed = _time.time() - _inference_start
                pct_display = 100 if is_last else int(pct)
                _log(f"[PROGRESS] Whisper進捗: {pct_display}% (チャンク {i+1}/{len(chunks)}) - {elapsed:.0f}/{audio_duration_sec:.0f}秒")
                if not is_last:
                    next_threshold = (int(pct // 20) + 1) * 20
                    next_threshold = min(next_threshold, 100)

            try:
                chunk_data = chunk_provider.get_chunk(i)
                if chunk_data is None or len(chunk_data) < 100:
                    continue

                inputs = processor(chunk_data, sampling_rate=16000, return_tensors="pt")  # type: ignore
                input_features = inputs.input_features.to(device).to(torch_dtype)  # type: ignore

                with torch.inference_mode():
                    generate_kwargs = {
                        "language": "ja",
                        "task": "transcribe",
                        "do_sample": False,
                        "temperature": 0.0,
                        "no_repeat_ngram_size": 0,
                        "use_cache": True,
                        "suppress_tokens": None,
                        "num_beams": 1,
                        "repetition_penalty": 1.0,
                    }
                    if prompt_ids is not None:
                        generate_kwargs["prompt_ids"] = prompt_ids

                    generated_tokens = model.generate(input_features, **generate_kwargs)  # pyright: ignore[union-attr]
                    generated_tokens = cast(torch.Tensor, generated_tokens)  # pyright: ignore

                transcription = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]  # type: ignore[union-attr]
                del generated_tokens, input_features, inputs
                if transcription.strip():
                    transcriptions.append(transcription.strip())
                del transcription, chunk_data

            except torch.cuda.OutOfMemoryError as e:
                _log(f"[ERROR] GPU メモリ不足 (チャンク {i+1}/{len(chunks)}): {e}", "error")
                chunk_errors += 1
                torch.cuda.empty_cache()
            except Exception as e:
                _log(f"[ERROR] チャンク {i+1}/{len(chunks)} 処理エラー: {e}", "error")
                chunk_errors += 1
                if chunk_errors > len(chunks) * 0.1:
                    raise RuntimeError(f"エラー率が高すぎます ({chunk_errors}/{len(chunks)}チャンク失敗)") from e

    del audio
    gc.collect()
    _log("[PROGRESS] PCMデータ解放完了")

    _inference_time = _time.time() - _inference_start
    _total_time = _time.time() - _preprocess_start
    error_part = "" if chunk_errors <= 0 else f" / エラー発生チャンク: {chunk_errors}/{len(chunks)}"
    _log(
        f"[PROGRESS] Whisper推論完了: {_inference_time:.1f}秒{error_part} / 総処理時間: {_total_time:.1f}秒 (前処理: {_preprocess_time:.1f}秒 + 推論: {_inference_time:.1f}秒) / 入力音声: {audio_duration_sec:.1f}秒 -> 処理速度: {audio_duration_sec/_total_time:.1f}x リアルタイム",
        "warning" if chunk_errors > 0 else "info",
    )

    full_transcription = "\n".join(transcriptions)
    if header:
        header_text = header.replace("\\n", "\n")
        full_transcription = header_text + "\n\n" + full_transcription

    output_file = Path(output_path) if output_path else audio_path_obj.with_suffix(".txt")
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_transcription)
        _log(f"=== テキストファイルに保存完了: {output_file} ===")
    except Exception as e:
        _log(f"ファイル保存エラー: {e}", "error")
        raise RuntimeError(f"出力ファイル保存に失敗: {output_file}") from e

    _log(f"=== 文字起こし完了: {audio_path_obj.name} ({len(full_transcription)}文字) ===")
    return full_transcription

# メイン実行
if __name__ == "__main__":
    _configure_stdio_for_windows()
    _register_signal_handlers()
    args = _parse_args()
    
    # 各音声ファイルを処理（終了時クリーンアップ保証）
    failed_files = []
    try:
        init_models()
        for i, audio_file in enumerate(args.audio_files):
            try:
                # 複数ファイル時は --output は使用不可（argparseでバリデーション済み）
                output_path = args.output if len(args.audio_files) == 1 else None
                transcribe_long_audio(
                    audio_file,
                    prompt=args.prompt,
                    header=args.header,
                    output_path=output_path,
                    vad_profile=args.vad_profile,
                )
            except Exception as e:  # type: ignore
                print(f"エラー: {audio_file} の処理中にエラーが発生しました: {e}")
                failed_files.append(audio_file)
    finally:
        _cleanup_resources()

    # 一つでも失敗があれば非ゼロで終了
    if failed_files:
        print(f"\n失敗したファイル ({len(failed_files)}件): {', '.join(failed_files)}")
        sys.exit(1)

"""
高速前処理パイプライン v2（Silero VAD 版）

=== 変更履歴 ===
v2.0 (2025-12): Silero VAD 導入による抜本的高速化
  - noisereduce 削除（Whisper自体のノイズ耐性に依存）
  - pydub detect_nonsilent → Silero VAD（〜100倍高速）
  - 500ms窓ループ削除（ベクトル演算に置換）

=== 設計思想 ===
1. Silero VAD による音声区間検出（音量非依存）
   - ニューラルネットワークで「人の声らしさ」を検出
   - 音量差に関係なく、大きな声の間の小さな声も検出
   - threshold=0.3 で高感度設定（デフォルト0.5より敏感）

2. RMS正規化（高速ベクトル演算）
   - 全体を-20dBFS相当に正規化
   - クリッピング防止

3. セグメント毎の音量保護
   - 検出した音声区間のRMSを計算
   - -35dB以下の小さな音声に軽微なダイナミックレンジ圧縮

4. 冒頭・末尾の取りこぼし対策
   - 最初の検出区間前と最後の検出区間後を自動追加

参考：
- Silero VAD: https://github.com/snakers4/silero-vad
- Whisper論文: https://arxiv.org/abs/2212.04356
"""
