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
from queue import Queue
from threading import Lock

# Windows cp932エンコーディングエラー回避: UTF-8で出力し、エンコード不可文字は置換
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ロガー設定: 呼び出し元から渡されたロガーを使用、なければ標準出力
_logger: Optional[logging.Logger] = None

def _log(msg: str, level: str = "info"):
    """ログ出力（logger があればそちらに、なければ print）"""
    if _logger:
        getattr(_logger, level)(msg)
    else:
        print(msg)

def set_logger(logger: logging.Logger):
    """外部からロガーを設定"""
    global _logger
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
    import cupy as cp  # type: ignore

    has_cupy = True  # type: ignore
    print("CuPy (GPU対応numpy) が利用可能です")
except ImportError:
    cp = np  # type: ignore
    has_cupy = False  # type: ignore
    print("CuPy が見つかりません。CPUでnumpyを使用します")
import torch
from pydub import AudioSegment  # type: ignore
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Silero VAD for fast voice activity detection
try:
    _vad_model, _vad_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True,
        verbose=False
    )
    _get_speech_timestamps, _, _read_audio, _, _ = _vad_utils
    has_silero_vad = True
    print("Silero VAD が利用可能です（高速音声検出）")
except Exception as e:
    has_silero_vad = False
    import traceback
    print(f"Silero VAD 読み込み失敗: {e}")
    print(traceback.format_exc())
    print("pydubフォールバックを使用します")
    from pydub.silence import detect_nonsilent  # type: ignore

_FLOAT_EPS = np.finfo(np.float32).eps


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
    """Silero VAD を5分割で実行し、20%刻みの進捗ログを可能にする。

    注意:
      - 境界の取りこぼしを避けるため overlap を入れてからマージする。
      - 返り値は全体の (start_ms, end_ms) に正規化済み。
    """
    if not has_silero_vad:
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
    next_pct = 100 if is_single_chunk else 20

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

        speech_timestamps = _get_speech_timestamps(
            audio_tensor,
            _vad_model,
            threshold=0.3,
            min_speech_duration_ms=100,
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
        del audio_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pct = int(round((i + 1) * 100 / chunk_count))
        while pct >= next_pct and next_pct <= 100:
            _log(f"[PROGRESS] VAD進捗: {next_pct}% ({i+1}/{chunk_count})")
            next_pct += 20

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
    """pydub の detect_nonsilent を5分割で実行し、20%刻みの進捗ログを可能にする。"""
    if "detect_nonsilent" not in globals():
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
    next_pct = 100 if is_single_chunk else 20

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

        pct = int(round((i + 1) * 100 / chunk_count))
        while pct >= next_pct and next_pct <= 100:
            _log(f"[PROGRESS] VAD進捗: {next_pct}% ({i+1}/{chunk_count})")
            next_pct += 20

    return _merge_ranges_ms(all_ranges, gap_ms=150)

model_id = "kotoba-tech/kotoba-whisper-v2.2"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# GPU利用可能なら Silero VAD モデルを GPU に移動
if has_silero_vad and torch.cuda.is_available():
    try:
        _vad_model = _vad_model.to(device)
        print("Silero VAD モデルを GPU に移動しました")
    except Exception as e:
        print(f"Silero VAD GPU移動失敗（CPU使用）: {e}")

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=torch_dtype, low_cpu_mem_usage=True, trust_remote_code=True).to(device)  # type: ignore
processor = AutoProcessor.from_pretrained(model_id)  # type: ignore

# === リソースクリーンアップ関連 ===
_CLEANED = False

def _cleanup_resources():
    """GPU/メモリ/モデルリソースを確実に解放する。
    強制終了（SIGINT/SIGTERM）や通常終了の両方で呼ばれる。
    """
    global model, processor, _CLEANED
    if _CLEANED:
        return
    print("[CLEANUP] 開始: リソース解放処理")
    # 推論モデル削除
    try:
        del model
    except Exception:
        pass
    try:
        del processor
    except Exception:
        pass
    # CuPy メモリプール解放
    try:
        if has_cupy and hasattr(cp, 'get_default_memory_pool'):
            cp.get_default_memory_pool().free_all_blocks()  # type: ignore
            print("[CLEANUP] CuPy メモリプール解放")
    except Exception as e:
        print(f"[CLEANUP] CuPy 解放失敗: {e}")
    # Torch GPU キャッシュ解放
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[CLEANUP] torch.cuda.empty_cache 実行")
    except Exception as e:
        print(f"[CLEANUP] torch キャッシュ解放失敗: {e}")
    # ガベージコレクション明示実行
    try:
        gc.collect()
        print("[CLEANUP] gc.collect 実行")
    except Exception:
        pass
    _CLEANED = True
    print("[CLEANUP] 完了")


def offload_model():
    """モデルをCPUに退避し、VRAMを解放する（他のアプリにGPUを譲るため）"""
    global model, _vad_model
    print("[RESOURCE] モデルをCPUに退避中...")
    try:
        if 'model' in globals() and model is not None:
            model.to("cpu")
    except Exception as e:
        print(f"Whisperモデル退避失敗: {e}")

    try:
        if has_silero_vad and 'vad_model' in globals() and _vad_model is not None:
            _vad_model.to("cpu")
    except Exception as e:
        # _vad_model might not be defined if init failed
        pass
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[RESOURCE] VRAM解放完了 (CPU待機状態)")


def restore_model():
    """モデルをGPUに復帰させる（処理開始前）"""
    global model, _vad_model
    if not torch.cuda.is_available():
        return

    # 既にGPUにあるか確認（簡易チェック）
    try:
        if 'model' in globals() and model is not None and next(model.parameters()).device.type == 'cuda':
            return
    except Exception:
        pass

    print("[RESOURCE] モデルをGPUに復帰中...")
    try:
        if 'model' in globals() and model is not None:
            model.to(device)
    except Exception as e:
        print(f"Whisperモデル復帰失敗: {e}")

    try:
        if has_silero_vad and 'vad_model' in globals() and _vad_model is not None:
            _vad_model.to(device)
    except Exception as e:
        pass
    print("[RESOURCE] GPU復帰完了")


def _signal_handler(signum, frame):
    """SIGINT/SIGTERM を捕捉して安全に終了"""
    print(f"[CLEANUP] シグナル受信: {signum}. クリーンアップを実行して終了します")
    _cleanup_resources()
    # シグナルコードを反映した終了ステータス (POSIX 慣例)
    try:
        sys.exit(128 + signum)
    except SystemExit:
        raise

# Windows 環境では SIGTERM が無い場合があるので存在確認
try:
    signal.signal(signal.SIGINT, _signal_handler)
except Exception:
    pass
try:
    signal.signal(signal.SIGTERM, _signal_handler)  # type: ignore
except Exception:
    pass

print(f"GPU利用可能: {torch.cuda.is_available()}")
print(f"transformers device: {device}")
if has_cupy and torch.cuda.is_available():
    print(f"CuPy GPU acceleration: 有効")
else:
    print(f"CuPy GPU acceleration: 無効 (CPU処理)")


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

    # Windows環境での日本語パス対策: 環境変数をUTF-8に強制
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, startupinfo=startupinfo, creationflags=creationflags, env=env)
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

    # Windows環境での日本語パス対策: 環境変数をUTF-8に強制
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, startupinfo=startupinfo, creationflags=creationflags, env=env)
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
    
    with av.open(path) as container:  # type: ignore
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

        resampler = av.audio.resampler.AudioResampler(  # type: ignore
            format="s16",
            layout="mono",
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

                frames = resampled if isinstance(resampled, list) else [resampled]
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
    with av.open(path) as container:  # type: ignore
        # 最初の音声ストリームを選択
        audio_stream = None
        for s in container.streams:  # type: ignore
            if s.type == "audio":  # type: ignore
                audio_stream = s
                break
        if audio_stream is None:
            raise ValueError("音声ストリームが見つかりませんでした")

        # 16kHz/mono/16bitPCMへリサンプル
        resampler = av.audio.resampler.AudioResampler(  # type: ignore
            format="s16",
            layout="mono",
            rate=sr,
        )

        pcm_chunks: list[np.ndarray] = []
        for packet in container.demux(audio_stream):  # type: ignore
            for frame in packet.decode():  # type: ignore
                # resampleの戻りは None / AudioFrame / [AudioFrame, ...] の場合がある
                resampled = resampler.resample(frame)  # type: ignore

                if resampled is None:
                    continue

                frames = resampled if isinstance(resampled, list) else [resampled]

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
                frames = flushed if isinstance(flushed, list) else [flushed]
                for f in frames:
                    buf = f.to_ndarray()  # type: ignore
                    if buf.ndim == 1:
                        pcm = buf.astype(np.int16)
                    else:
                        pcm = buf.astype(np.int16).mean(axis=0).astype(np.int16)
                    if pcm.size:
                        pcm_chunks.append(pcm)
        except Exception:
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
    
    def __init__(self, audio_file: str, chunks: list, prefetch_count: int = 3):
        """
        Args:
            audio_file: 音声/動画ファイルパス
            chunks: (start_ms, end_ms) のリスト
            prefetch_count: 先読みするチャンク数（デフォルト3）
        """
        self.audio_file = audio_file
        self.chunks = chunks
        self.prefetch_count = prefetch_count
        
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
            self._executor.shutdown(wait=False)
            self._executor = None
        if self._container:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
        return False
    
    def _open_pyav_container(self):
        """PyAVコンテナを開く（1回のみ）"""
        if not has_pyav:
            return
        try:
            self._container = av.open(self.audio_file)  # type: ignore
            for s in self._container.streams:  # type: ignore
                if s.type == "audio":  # type: ignore
                    self._audio_stream = s
                    break
            if self._audio_stream:
                self._resampler = av.audio.resampler.AudioResampler(  # type: ignore
                    format="s16",
                    layout="mono",
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
                
                # 新しいresamplerを作成（シーク後に状態リセット）
                resampler = av.audio.resampler.AudioResampler(  # type: ignore
                    format="s16",
                    layout="mono",
                    rate=16000,
                )
                
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
                        
                        frames = resampled if isinstance(resampled, list) else [resampled]
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
            error_msg = str(e) if e else type(e).__name__
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
                result = future.result(timeout=30)
                # _load_chunk は (idx, data) を返す。data が None なら読み込み失敗
                _, data = result
                if data is not None:
                    return data
                # data=None: プリフェッチでの読み込み失敗 → 警告のみ（再試行しない）
                _log(f"Prefetch chunk {chunk_idx} returned None data", "warning")
                return None
            except Exception as e:
                error_msg = str(e) if str(e) else type(e).__name__
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
) -> str:
    """
    長い音声を分割して処理し、結果を連結する
    
    Args:
        audio_file: 音声/動画ファイルのパス
        prompt: AIに渡すプロンプト文字列（任意）
        header: 出力ファイルの冒頭に付加するヘッダ（任意）
        output_path: 出力ファイルパス（Noneの場合は入力ファイルと同じ場所に .txt）
    
    Returns:
        文字起こし結果のテキスト
        
    Raises:
        FileNotFoundError: 入力ファイルが存在しない場合
        RuntimeError: 文字起こし処理中にエラーが発生した場合
    """
    import time as _time
    
    # 入力ファイル存在確認
    audio_path_obj = Path(audio_file)
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"入力ファイルが存在しません: {audio_file}")
    
    _log(f"=== 文字起こし開始: {audio_path_obj.name} ===")
    
    if prompt:
        _log(f"プロンプト: {prompt}")

    # 音声を読み込み（mkvなど動画コンテナは"音声のみ"抽出）
    try:
        ext = Path(audio_file).suffix.lower()
        is_video_container = ext in {".mkv", ".mp4", ".webm", ".mov", ".m4v", ".flv", ".avi", ".wmv"}
        if is_video_container and has_pyav:
            _log("PyAV経由で音声ストリームのみを読み込みます (in-process, 16kHz/mono)")
            try:
                audio = load_audio_via_pyav(audio_file, sr=16000)
            except Exception as e:
                _log(f"PyAV読み込み失敗: {e}", "warning")
                if _ffmpeg_available():
                    _log("ffmpegサブプロセスにフォールバックします (-vn, 16kHz/mono)")
                    audio = load_audio_via_ffmpeg(audio_file, sr=16000)
                else:
                    _log("librosaにフォールバックします")
                    audio, _ = librosa.load(audio_file, sr=16000)  # type: ignore
        elif is_video_container and _ffmpeg_available():
            _log("ffmpeg経由で音声ストリームのみを読み込みます (-vn, 16kHz/mono)")
            audio = load_audio_via_ffmpeg(audio_file, sr=16000)
        else:
            # 非対応拡張子やffmpeg無しの場合はlibrosaにフォールバック
            audio, _ = librosa.load(audio_file, sr=16000)  # type: ignore
    except Exception as e:
        _log(f"音声読み込みエラー: {e}", "error")
        raise RuntimeError(f"音声読み込みに失敗しました: {audio_file}") from e

    # === 高速前処理パイプライン（Silero VAD + 軽量正規化） ===
    audio_duration_sec = len(audio) / 16000
    _log(f"[PROGRESS] 音声読み込み完了: {audio_duration_sec:.1f}秒 ({audio_duration_sec/60:.1f}分)")
    _log(f"[PROGRESS] 前処理開始... (GPU: {torch.cuda.is_available()}, デバイス: {device})")
    _preprocess_start = _time.time()

    # 1. NaN/Inf をクリーンアップ（必須）
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

    # 2. 高速RMS正規化（ベクトル演算）
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        target_rms = 0.1  # 約-20dBFS
        audio = audio * (target_rms / rms)
        audio = np.clip(audio, -1.0, 1.0)
    audio = cast(np.ndarray, audio)

    # 3. Silero VAD で高速音声区間検出（音量非依存、小さな声も検出）
    _vad_start = _time.time()
    
    if has_silero_vad:
        if torch.cuda.is_available():
            _log(f"[PROGRESS] Silero VAD 実行中 (GPU: {device})...")
        else:
            _log("[PROGRESS] Silero VAD 実行中 (CPU)...")

        # 20%刻みの進捗を出すため、全音声を5分割してVAD実行 → 範囲をマージ
        nonsilent_ranges = _silero_vad_in_chunks(audio, sr=16000, chunk_count=5, overlap_ms=300)
        
        _vad_time = _time.time() - _vad_start
        _log(f"[PROGRESS] Silero VAD 完了: {len(nonsilent_ranges)} セグメント検出, {_vad_time:.2f}秒")
        
        # GPU メモリ解放は _silero_vad_in_chunks 側で実施
    else:
        # フォールバック: pydub の detect_nonsilent（遅い）
        _log("[PROGRESS] 警告: Silero VAD 利用不可 -> pydub (低速)", "warning")
        nonsilent_ranges = _pydub_vad_in_chunks(
            audio,
            sr=16000,
            chunk_count=5,
            overlap_ms=300,
            min_silence_len=400,
            silence_thresh=-50,
            seek_step=25,
        )
        _log(f"pydub: {len(nonsilent_ranges)} 個の音声セグメントを検出")

    # === VAD完了: audio はインメモリチャンクプロバイダで再利用 ===
    audio_length_ms = int(audio_duration_sec * 1000)
    # 注意: audio は削除しない（InMemoryChunkProvider で再利用）
    _log(f"[PROGRESS] VAD完了 (PCMデータ保持: {len(audio) * 4 / 1024 / 1024:.1f}MB)")

    # 6. 冒頭・末尾の取りこぼし対策 + 長時間セグメント分割
    max_chunk_ms = 25000  # 25秒（Whisperに適したサイズ）
    processed_ranges = []

    # 冒頭対策
    if nonsilent_ranges and nonsilent_ranges[0][0] > 500:
        processed_ranges.append((0, nonsilent_ranges[0][0]))
        _log(f"冒頭追加: 0-{nonsilent_ranges[0][0]}ms")

    # 各セグメントを処理
    for start_ms, end_ms in nonsilent_ranges:
        duration = end_ms - start_ms
        if duration > 20000:
            # 長時間セグメントは15秒ずつに分割（3秒オーバーラップ）
            current = start_ms
            while current < end_ms:
                seg_end = min(current + 15000, end_ms)
                if seg_end - current >= 500:
                    processed_ranges.append((current, seg_end))
                current += 12000
        else:
            processed_ranges.append((start_ms, end_ms))

    # 末尾対策
    if nonsilent_ranges and nonsilent_ranges[-1][1] < audio_length_ms - 500:
        processed_ranges.append((nonsilent_ranges[-1][1], audio_length_ms))
        _log(f"末尾追加: {nonsilent_ranges[-1][1]}-{audio_length_ms}ms")

    # 7. チャンクにまとめる（隣接セグメントを結合）
    chunks = []
    current_start = None
    current_end = None

    for start_ms, end_ms in processed_ranges:
        if current_start is None:
            current_start = start_ms
            current_end = end_ms
        elif (end_ms - current_start) <= max_chunk_ms:
            current_end = end_ms
        else:
            chunks.append((current_start, current_end))
            current_start = start_ms
            current_end = end_ms

    if current_start is not None:
        chunks.append((current_start, current_end))

    _preprocess_time = _time.time() - _preprocess_start
    total_speech_sec = sum((e - s) / 1000 for s, e in chunks)
    _log(
        f"[PROGRESS] 前処理完了: {_preprocess_time:.2f}秒 / チャンク数: {len(chunks)} / 音声合計: {total_speech_sec:.1f}秒 / Whisper推論開始"
    )
    _inference_start = _time.time()

    # プロンプトからprompt_idsを生成
    prompt_ids = None
    if prompt:
        try:
            # Whisper tokenizer でプロンプトをトークン化
            # transformers 4.57+では prompt_ids は Rank-1 テンソル（1次元）が必要
            prompt_ids = processor.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")  # type: ignore
            prompt_ids = prompt_ids.squeeze(0).to(device)  # 2次元→1次元に変換してからGPUへ
            _log(f"prompt_ids shape: {prompt_ids.shape}")  # type: ignore
        except Exception as e:
            _log(f"プロンプトトークン化エラー(無視して続行): {e}", "warning")
            prompt_ids = None

    transcriptions = []  # type: ignore
    chunk_errors = 0  # エラーカウント
    
    # インメモリチャンクプロバイダ: VAD で読み込んだ PCM を再利用（I/O ゼロ）
    with InMemoryChunkProvider(audio, chunks) as chunk_provider:
        _log(f"[PROGRESS] チャンク読み込みモード: {chunk_provider.mode}")

        if len(chunks) == 0:
            _log("[PROGRESS] Whisper進捗: 100% (チャンク 0/0)")
        else:
            next_pct = 20

        for i in range(len(chunks)):
            if len(chunks) > 0:
                pct_int = int(round((i + 1) * 100 / len(chunks)))
                if i == len(chunks) - 1:
                    pct_int = 100

                while pct_int >= next_pct and next_pct <= 100:
                    elapsed = _time.time() - _inference_start
                    _log(
                        f"[PROGRESS] Whisper進捗: {next_pct}% (チャンク {i+1}/{len(chunks)}) - {elapsed:.0f}/{audio_duration_sec:.0f}秒"
                    )
                    next_pct += 20

            try:
                # メモリからチャンクを取得（I/O ゼロ）
                chunk_data = chunk_provider.get_chunk(i)
                
                if chunk_data is None or len(chunk_data) < 100:
                    continue

                # 音声を処理
                inputs = processor(chunk_data, sampling_rate=16000, return_tensors="pt")  # type: ignore
                input_features = inputs.input_features.to(device).to(torch_dtype)  # type: ignore

                with torch.no_grad():
                    # generate 引数を構築
                    generate_kwargs = {
                        "language": "ja",
                        "task": "transcribe",
                        "do_sample": False,
                        "temperature": 0.0,
                        "no_repeat_ngram_size": 0,
                        "use_cache": True,
                        "suppress_tokens": None,  # 抑制トークンを無効化
                        "num_beams": 1,  # ビームサーチを無効化して感度向上
                        "repetition_penalty": 1.0,  # 繰り返しペナルティを無効化
                    }
                    
                    # プロンプトがある場合は prompt_ids を追加
                    if prompt_ids is not None:
                        generate_kwargs["prompt_ids"] = prompt_ids
                    
                    generated_tokens = model.generate(  # pyright: ignore
                        input_features,
                        **generate_kwargs
                    )
                    generated_tokens = cast(torch.Tensor, generated_tokens)  # pyright: ignore

                transcription: str = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]  # type: ignore

                if transcription.strip():  # type: ignore  # 空でない場合のみ追加
                    transcriptions.append(transcription.strip())  # type: ignore
                    
            except torch.cuda.OutOfMemoryError as e:
                # CUDA OOM: 致命的なエラーとして報告（復旧不能の可能性）
                _log(f"[ERROR] GPU メモリ不足 (チャンク {i+1}/{len(chunks)}): {e}", "error")
                _log("[ERROR] GPU メモリを解放して再試行を推奨", "error")
                chunk_errors += 1
                torch.cuda.empty_cache()  # 試しにキャッシュをクリア
            except Exception as e:
                # その他のエラー: ログして続行
                _log(f"[ERROR] チャンク {i+1}/{len(chunks)} 処理エラー: {e}", "error")
                chunk_errors += 1
                if chunk_errors > len(chunks) * 0.1:  # 10%以上失敗したら中断
                    raise RuntimeError(f"エラー率が高すぎます ({chunk_errors}/{len(chunks)}チャンク失敗)") from e

    # === メモリ解放: 推論完了後に PCM データを解放 ===
    del audio
    gc.collect()
    _log(f"[PROGRESS] PCMデータ解放完了")

    # 推論完了サマリー
    _inference_time = _time.time() - _inference_start
    _total_time = _time.time() - _preprocess_start
    error_part = "" if chunk_errors <= 0 else f" / エラー発生チャンク: {chunk_errors}/{len(chunks)}"
    _log(
        f"[PROGRESS] Whisper推論完了: {_inference_time:.1f}秒{error_part} / 総処理時間: {_total_time:.1f}秒 (前処理: {_preprocess_time:.1f}秒 + 推論: {_inference_time:.1f}秒) / 入力音声: {audio_duration_sec:.1f}秒 -> 処理速度: {audio_duration_sec/_total_time:.1f}x リアルタイム",
        "warning" if chunk_errors > 0 else "info",
    )

    # 全てのチャンクの結果を連結（改行区切り）
    full_transcription = "\n".join(transcriptions)  # type: ignore

    # ヘッダを付加（\\n を実際の改行に変換）
    if header:
        header_text = header.replace('\\n', '\n')
        full_transcription = header_text + "\n\n" + full_transcription

    # テキストファイルに出力
    audio_path = Path(audio_file)
    if output_path:
        output_file = Path(output_path)
    else:
        output_file = audio_path.with_suffix('.txt')

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_transcription)
        _log(f"=== テキストファイルに保存完了: {output_file} ===")
    except Exception as e:
        _log(f"ファイル保存エラー: {e}", "error")
        raise RuntimeError(f"出力ファイル保存に失敗: {output_file}") from e

    _log(f"=== 文字起こし完了: {audio_path_obj.name} ({len(full_transcription)}文字) ===")

    return full_transcription  # type: ignore


# メイン実行
if __name__ == "__main__":
    args = _parse_args()
    
    # 各音声ファイルを処理（終了時クリーンアップ保証）
    failed_files = []
    try:
        for i, audio_file in enumerate(args.audio_files):
            try:
                # 複数ファイル時は --output は使用不可（argparseでバリデーション済み）
                output_path = args.output if len(args.audio_files) == 1 else None
                transcribe_long_audio(
                    audio_file,
                    prompt=args.prompt,
                    header=args.header,
                    output_path=output_path,
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
