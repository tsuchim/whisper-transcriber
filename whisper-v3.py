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
import subprocess
import shutil
from pathlib import Path
from typing import cast, Optional
import signal
import gc
import io
import argparse

# Windows cp932エンコーディングエラー回避: UTF-8で出力し、エンコード不可文字は置換
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
from pydub.effects import compress_dynamic_range  # type: ignore
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
    print(f"Silero VAD 読み込み失敗: {e}（pydubフォールバック）")
    from pydub.silence import detect_nonsilent  # type: ignore

_FLOAT_EPS = np.finfo(np.float32).eps

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

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, startupinfo=startupinfo, creationflags=creationflags)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm.size == 0:
        raise ValueError("ffmpegで音声データを取得できませんでした")
    audio = pcm.astype(np.float32) / 32767.0
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
    """
    print(f"\n=== {audio_file} ===")
    
    if prompt:
        print(f"プロンプト: {prompt}")

    # 音声を読み込み（mkvなど動画コンテナは"音声のみ"抽出）
    ext = Path(audio_file).suffix.lower()
    is_video_container = ext in {".mkv", ".mp4", ".webm", ".mov", ".m4v", ".flv", ".avi", ".wmv"}
    if is_video_container and has_pyav:
        print("PyAV経由で音声ストリームのみを読み込みます (in-process, 16kHz/mono)")
        try:
            audio = load_audio_via_pyav(audio_file, sr=16000)
        except Exception as e:
            print(f"PyAV読み込み失敗: {e}")
            if _ffmpeg_available():
                print("ffmpegサブプロセスにフォールバックします (-vn, 16kHz/mono)")
                audio = load_audio_via_ffmpeg(audio_file, sr=16000)
            else:
                print("librosaにフォールバックします")
                audio, _ = librosa.load(audio_file, sr=16000)  # type: ignore
    elif is_video_container and _ffmpeg_available():
        print("ffmpeg経由で音声ストリームのみを読み込みます (-vn, 16kHz/mono)")
        audio = load_audio_via_ffmpeg(audio_file, sr=16000)
    else:
        # 非対応拡張子やffmpeg無しの場合はlibrosaにフォールバック
        audio, _ = librosa.load(audio_file, sr=16000)  # type: ignore

    # === 高速前処理パイプライン（Silero VAD + 軽量正規化） ===
    import time as _time
    audio_duration_sec = len(audio) / 16000
    print(f"[PROGRESS] 音声読み込み完了: {audio_duration_sec:.1f}秒 ({audio_duration_sec/60:.1f}分)")
    print(f"[PROGRESS] 前処理開始... (GPU: {torch.cuda.is_available()}, デバイス: {device})")
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
        # Silero VAD は torch.Tensor を期待
        audio_tensor = torch.from_numpy(audio).float()
        
        # GPU利用可能なら GPU に移動
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.to(device)
            print(f"[PROGRESS] Silero VAD 実行中 (GPU: {device})...")
        else:
            print("[PROGRESS] Silero VAD 実行中 (CPU)...")
        
        # 高感度設定: threshold を下げて小さな声も漏れなく検出
        speech_timestamps = _get_speech_timestamps(
            audio_tensor,
            _vad_model,
            threshold=0.3,               # デフォルト0.5→0.3で感度UP
            min_speech_duration_ms=100,  # 100ms以上の発話を検出
            min_silence_duration_ms=150, # 150msの無音で分割（細かく）
            speech_pad_ms=100,           # 発話前後に100msのマージン
            return_seconds=False,        # サンプル単位で返す
        )
        
        # サンプル単位 → ミリ秒単位に変換
        nonsilent_ranges = []
        for ts in speech_timestamps:
            start_ms = int(ts['start'] / 16)  # 16kHz → ms
            end_ms = int(ts['end'] / 16)
            nonsilent_ranges.append((start_ms, end_ms))
        
        _vad_time = _time.time() - _vad_start
        print(f"[PROGRESS] Silero VAD 完了: {len(nonsilent_ranges)} セグメント検出, {_vad_time:.2f}秒")
    else:
        # フォールバック: pydub の detect_nonsilent（遅い）
        print("[PROGRESS] 警告: Silero VAD 利用不可 → pydub (低速)")
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_segment_fallback = AudioSegment(audio_int16.tobytes(), frame_rate=16000, sample_width=2, channels=1)
        nonsilent_ranges = detect_nonsilent(
            audio_segment_fallback,
            min_silence_len=400,
            silence_thresh=-50,
            seek_step=25,
        )
        print(f"pydub: {len(nonsilent_ranges)} 個の音声セグメントを検出")

    # 4. pydubでの処理のためにAudioSegmentに変換
    audio_int16_cpu = (audio * 32767).astype(np.int16)
    audio_segment = AudioSegment(audio_int16_cpu.tobytes(), frame_rate=16000, sample_width=2, channels=1)
    audio_length_ms = len(audio_segment)

    # 5. セグメント毎の音量分析と小さな音声の増幅（高速ベクトル演算）
    ffmpeg_ready = _ffmpeg_available()
    
    if nonsilent_ranges:
        # 各セグメントのRMSを高速計算
        segment_rms_list = []
        for start_ms, end_ms in nonsilent_ranges:
            start_sample = int(start_ms * 16)
            end_sample = min(int(end_ms * 16), len(audio))
            if end_sample > start_sample:
                seg_audio = audio[start_sample:end_sample]
                seg_rms = np.sqrt(np.mean(seg_audio**2))
                if seg_rms > 0:
                    seg_db = 20 * np.log10(seg_rms)
                    segment_rms_list.append(seg_db)
        
        if segment_rms_list:
            min_seg_db = min(segment_rms_list)
            max_seg_db = max(segment_rms_list)
            print(f"セグメント音量: 最小={min_seg_db:.1f}dB, 最大={max_seg_db:.1f}dB, 差={max_seg_db-min_seg_db:.1f}dB")
            
            # 小さな音声セグメントがある場合の保護処理
            if min_seg_db < -35 and ffmpeg_ready:
                audio_segment = compress_dynamic_range(
                    audio_segment, threshold=-35.0, ratio=3.0, attack=5.0, release=50.0
                )
                print(f"小さな音声保護: 軽微な圧縮を適用（{min_seg_db:.1f}dB）")

    # 6. 冒頭・末尾の取りこぼし対策 + 長時間セグメント分割
    max_chunk_ms = 25000  # 25秒（Whisperに適したサイズ）
    processed_ranges = []

    # 冒頭対策
    if nonsilent_ranges and nonsilent_ranges[0][0] > 500:
        processed_ranges.append((0, nonsilent_ranges[0][0]))
        print(f"冒頭追加: 0-{nonsilent_ranges[0][0]}ms")

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
        print(f"末尾追加: {nonsilent_ranges[-1][1]}-{audio_length_ms}ms")

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
    print(f"[PROGRESS] ========================================")
    print(f"[PROGRESS] 前処理完了: {_preprocess_time:.2f}秒")
    print(f"[PROGRESS] チャンク数: {len(chunks)}, 音声合計: {total_speech_sec:.1f}秒")
    print(f"[PROGRESS] Whisper推論開始...")
    print(f"[PROGRESS] ========================================")
    _inference_start = _time.time()

    # プロンプトからprompt_idsを生成
    prompt_ids = None
    if prompt:
        try:
            # Whisper tokenizer でプロンプトをトークン化
            # transformers 4.57+では prompt_ids は Rank-1 テンソル（1次元）が必要
            prompt_ids = processor.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")  # type: ignore
            prompt_ids = prompt_ids.squeeze(0).to(device)  # 2次元→1次元に変換してからGPUへ
            print(f"prompt_ids shape: {prompt_ids.shape}")  # type: ignore
        except Exception as e:
            print(f"プロンプトトークン化エラー（無視して続行）: {e}")
            prompt_ids = None

    transcriptions = []  # type: ignore

    for i, (start_ms, end_ms) in enumerate(chunks):  # type: ignore
        pct = (i + 1) / len(chunks) * 100
        if i % 10 == 0 or i == len(chunks) - 1:  # 10チャンク毎 + 最後
            elapsed = _time.time() - _inference_start
            print(f"[PROGRESS] チャンク {i+1}/{len(chunks)} ({pct:.0f}%) - 経過: {elapsed:.1f}秒")  # type: ignore

        # チャンクを抽出
        chunk_segment = audio_segment[start_ms:end_ms]  # type: ignore

        # 短すぎるチャンクは処理しない（より短いチャンクも処理）
        if len(chunk_segment) < 200:  # type: ignore
            continue

        # numpy配列に変換（GPU対応）
        # Extract samples as numpy array on CPU. Keeping this on host memory avoids
        # repeated GPU->CPU round trips which were present with cp.asnumpy.
        chunk_data = np.array(chunk_segment.get_array_of_samples(), dtype=np.float32) / 32767.0  # type: ignore

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

    # 推論完了サマリー
    _inference_time = _time.time() - _inference_start
    _total_time = _time.time() - _preprocess_start
    print(f"[PROGRESS] ========================================")
    print(f"[PROGRESS] Whisper推論完了: {_inference_time:.1f}秒")
    print(f"[PROGRESS] 総処理時間: {_total_time:.1f}秒 (前処理: {_preprocess_time:.1f}秒 + 推論: {_inference_time:.1f}秒)")
    print(f"[PROGRESS] 入力音声: {audio_duration_sec:.1f}秒 → 処理速度: {audio_duration_sec/_total_time:.1f}x リアルタイム")
    print(f"[PROGRESS] ========================================")

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
        print(f"\n=== テキストファイルに保存完了 ===")
        print(f"出力先: {output_file}")
    except Exception as e:
        print(f"ファイル保存エラー: {e}")
        raise  # Re-raise to ensure non-zero exit code

    print(f"\n=== 完全な転写結果 ===")  # type: ignore
    print(full_transcription)  # type: ignore

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
